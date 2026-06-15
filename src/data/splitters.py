import math
import random
from itertools import combinations, islice

import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta
from src.config import IS_MONTHS, OOS_MONTHS, RANDOM_STATE
from src.visualization.style import DEFAULT_STYLE, PlotStyle


def split_train_test_live(
    df: pd.DataFrame,
    train_end: str | pd.Timestamp,
    test_end: str | pd.Timestamp,
    date_index_col: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Deterministically split a time-indexed DataFrame into train/test/live windows.

    Boundaries are inclusive on the left windows:
    - train: index <= train_end
    - test: train_end < index <= test_end
    - live: index > test_end
    """
    if df.empty:
        return df.copy(), df.copy(), df.copy()

    idx = pd.to_datetime(df.index) if date_index_col is None else pd.to_datetime(df[date_index_col])

    idx = pd.DatetimeIndex(idx)
    train_end_ts = pd.Timestamp(train_end)
    test_end_ts = pd.Timestamp(test_end)

    if idx.tz is not None:
        train_end_ts = train_end_ts.tz_localize(idx.tz) if train_end_ts.tz is None else train_end_ts.tz_convert(idx.tz)

        test_end_ts = test_end_ts.tz_localize(idx.tz) if test_end_ts.tz is None else test_end_ts.tz_convert(idx.tz)
    else:
        train_end_ts = train_end_ts.tz_localize(None) if train_end_ts.tz is not None else train_end_ts
        test_end_ts = test_end_ts.tz_localize(None) if test_end_ts.tz is not None else test_end_ts

    if test_end_ts <= train_end_ts:
        raise ValueError("test_end must be strictly greater than train_end.")

    train_mask = idx <= train_end_ts
    test_mask = (idx > train_end_ts) & (idx <= test_end_ts)
    live_mask = idx > test_end_ts

    return df.loc[train_mask].copy(), df.loc[test_mask].copy(), df.loc[live_mask].copy()


# ==============================
# Month-based Walk-Forward Splitter
# ==============================
class MonthWalkForwardSplitter:
    def __init__(
        self,
        offset_months: int = 0,
        anchored: bool = False,
        embargo_months: int = 0,
        market_tz: str = "Europe/Berlin",
        is_months: int = IS_MONTHS,
        oos_months: int = OOS_MONTHS,
    ) -> None:
        self.is_months = is_months
        self.oos_months = oos_months
        self.offset_months = int(offset_months)
        self.anchored = bool(anchored)
        self.embargo_months = int(embargo_months)
        self.market_tz = market_tz

    @staticmethod
    def _global_month(idx: pd.DatetimeIndex) -> tuple[np.ndarray, int, int]:
        y = idx.year
        m = idx.month
        y0 = int(y.min())
        g = (y - y0) * 12 + m
        return g.astype(np.int32), int(g.min()), int(g.max())

    @staticmethod
    def _mask_by_month_range(gcodes: np.ndarray, start_m: int, end_m: int) -> np.ndarray:
        return (gcodes >= start_m) & (gcodes <= end_m)

    def _ensure_tz(self, idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
        if idx.tz is None:
            return idx.tz_localize(self.market_tz)
        return idx.tz_convert(self.market_tz)

    def split(
        self, df: pd.DataFrame, *, date_index_col: str | None = None
    ) -> tuple[list[pd.DataFrame], list[pd.DataFrame], pd.DataFrame]:
        idx = pd.to_datetime(df.index) if date_index_col is None else pd.to_datetime(df[date_index_col])
        idx = self._ensure_tz(pd.DatetimeIndex(idx))

        g, gmin, gmax = self._global_month(idx)

        train_dfs: list[pd.DataFrame] = []
        test_dfs: list[pd.DataFrame] = []

        # Initial test nominal month with clamp to minimum viable
        min_first_test_nominal = gmin + self.is_months
        current_test_nominal = min_first_test_nominal + self.offset_months
        current_test_nominal = max(current_test_nominal, min_first_test_nominal)

        last_test_end_code: int | None = None

        while True:
            test_end = min(current_test_nominal + self.oos_months - 1, gmax)
            test_start = current_test_nominal + self.embargo_months

            if test_start > test_end:
                break

            train_end = current_test_nominal - 1

            train_start = gmin if self.anchored else current_test_nominal - self.is_months

            if train_end < train_start:
                current_test_nominal += self.oos_months
                continue

            mask_train = self._mask_by_month_range(g, train_start, train_end)
            mask_test = self._mask_by_month_range(g, test_start, test_end)

            train_dfs.append(df.loc[mask_train].copy())
            test_dfs.append(df.loc[mask_test].copy())

            last_test_end_code = test_end
            current_test_nominal += self.oos_months

        latest_df = pd.DataFrame()
        if last_test_end_code is not None:
            latest_start = max(gmin, last_test_end_code - self.is_months + 1)
            latest_end = last_test_end_code
            mask_latest = self._mask_by_month_range(g, latest_start, latest_end)
            latest_df = df.loc[mask_latest].copy()

        return train_dfs, test_dfs, latest_df

    @staticmethod
    def print_train_test_periods(train_dfs, test_dfs) -> None:
        print(f"Number of train-test splits: {len(train_dfs)}")
        for i, (tr, te) in enumerate(zip(train_dfs, test_dfs, strict=True), 1):
            if len(tr) == 0 or len(te) == 0:
                print(f"Set {i}: (empty split)")
                continue
            ts = pd.to_datetime(tr.index.min()).strftime("%Y-%m")
            te_ = pd.to_datetime(tr.index.max()).strftime("%Y-%m")
            xs = pd.to_datetime(te.index.min()).strftime("%Y-%m")
            xe = pd.to_datetime(te.index.max()).strftime("%Y-%m")
            print(f"Set {i}: Train = {ts} to {te_}, Test = {xs} to {xe}")

    def get_latest_train(
        self, df: pd.DataFrame, *, date_index_col: str | None = None, mode: str = "last_is_window"
    ) -> pd.DataFrame:
        idx = pd.to_datetime(df.index) if date_index_col is None else pd.to_datetime(df[date_index_col])
        idx = self._ensure_tz(pd.DatetimeIndex(idx))
        g, gmin, gmax = self._global_month(idx)

        if mode == "last_is_window":
            train_start = gmax - self.is_months + 1
            train_start = max(train_start, gmin)
            train_end = gmax
            mask = self._mask_by_month_range(g, train_start, train_end)
            return df.loc[mask].copy()
        elif mode == "expand_to_latest":
            mask = self._mask_by_month_range(g, gmin, gmax)
            return df.loc[mask].copy()
        else:
            raise ValueError(f"Unknown mode '{mode}'. Valid modes: 'last_is_window', 'expand_to_latest'.")


# ==============================
# CPCV: Purged & Embargoed Combinatorial Paths
# ==============================
class CPCVSplitter:
    def __init__(
        self,
        test_months: int | None = None,
        S: int | None = None,
        purged_weeks: int = 4,
        embargo_weeks: int = 4,
        max_paths_to_return: int | None = 20,
        randomly_sample_paths: bool = True,
        random_state: int | None = RANDOM_STATE,
        market_tz: str = "Europe/Berlin",
        target_train_size_pct: float = 0.7,
        target_n_test_paths: int = 20,
        weight_train_size: float = 1.0,
        weight_n_test_paths: float = 1.0,
        max_folds: int = 100,
        date_col: str | None = None,
        holdout_pct: float = 0.0,
    ) -> None:
        self.test_months = test_months
        self.S = S
        self.purged_weeks = int(purged_weeks)
        self.embargo_weeks = int(embargo_weeks)
        self.max_paths_to_return = max_paths_to_return
        self.randomly_sample_paths = bool(randomly_sample_paths)
        self.random_state = random_state
        self.market_tz = market_tz
        self.target_train_size_pct = float(target_train_size_pct)
        self.target_n_test_paths = int(target_n_test_paths)
        self.weight_train_size = float(weight_train_size)
        self.weight_n_test_paths = float(weight_n_test_paths)
        self.max_folds = int(max_folds)
        self.date_col = date_col
        self.holdout_pct = float(holdout_pct)

    def _ensure_tz(self, idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
        if idx.tz is None:
            return idx.tz_localize(self.market_tz)
        return idx.tz_convert(self.market_tz)

    def _resolve_index(self, df: pd.DataFrame) -> pd.DatetimeIndex:
        if self.date_col is None:
            date_values = df.index
        elif isinstance(self.date_col, str):
            if self.date_col in df.columns:
                date_values = df[self.date_col]
            elif df.index.name == self.date_col:
                date_values = df.index
            else:
                raise KeyError(f"date_col '{self.date_col}' not found in DataFrame columns or index name.")
        else:
            # Handle accidental non-string inputs (e.g., passing df.index).
            date_values = df.index

        idx = pd.to_datetime(date_values, utc=True)
        return self._ensure_tz(pd.DatetimeIndex(idx))

    def _sort_df(self, df: pd.DataFrame) -> pd.DataFrame:
        if isinstance(self.date_col, str) and self.date_col in df.columns:
            return df.sort_values(self.date_col)
        return df.sort_index()

    def _get_total_months(self, df: pd.DataFrame) -> int:
        if df.empty:
            return 0
        idx = self._resolve_index(df)
        min_date = idx.min()
        max_date = idx.max()
        return (max_date.year - min_date.year) * 12 + (max_date.month - min_date.month) + 1

    def _build_month_groups(self, df: pd.DataFrame) -> tuple[list[pd.Timestamp], list[pd.Timestamp]]:
        if df.empty:
            return [], []
        idx = self._resolve_index(df).sort_values()
        tz = idx.tz
        min_date, max_date = idx.min(), idx.max()
        month_start = pd.Timestamp(year=min_date.year, month=min_date.month, day=1, tz=tz)
        starts, ends = [], []
        current = month_start
        while current + relativedelta(months=self.test_months) <= max_date:
            starts.append(current)
            ends.append(current + relativedelta(months=self.test_months))
            current = current + relativedelta(months=self.test_months)
        return starts, ends

    @staticmethod
    def _merge_intervals(intervals: list[tuple[pd.Timestamp, pd.Timestamp]]) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
        if not intervals:
            return []
        intervals = [(s, e) for s, e in intervals if s < e]
        if not intervals:
            return []
        intervals = sorted(intervals, key=lambda x: x[0])
        merged = [intervals[0]]
        for start, end in intervals[1:]:
            last_start, last_end = merged[-1]
            if start <= last_end:
                merged[-1] = (last_start, max(last_end, end))
            else:
                merged.append((start, end))
        return merged

    @staticmethod
    def _mask_include_from_exclusions(
        idx: pd.DatetimeIndex, exclusions: list[tuple[pd.Timestamp, pd.Timestamp]]
    ) -> np.ndarray:
        include = np.ones(len(idx), dtype=bool)
        for start, end in exclusions:
            i = idx.searchsorted(start, side="left")
            j = idx.searchsorted(end, side="left")
            if i < j:
                include[i:j] = False
        return include

    def _build_exclusion_intervals_for_path(
        self,
        comb: tuple[int, ...],
        group_starts: list[pd.Timestamp],
        group_ends: list[pd.Timestamp],
        purge_delta: pd.Timedelta,
        embargo_delta: pd.Timedelta,
    ) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
        ex: list[tuple[pd.Timestamp, pd.Timestamp]] = []
        for i, g in enumerate(comb):
            test_start, test_end = group_starts[g], group_ends[g]
            is_prev_adjacent = i > 0 and group_starts[comb[i]] == group_ends[comb[i - 1]]
            is_next_adjacent = i < len(comb) - 1 and group_starts[comb[i + 1]] == test_end
            purge_before_start = (
                test_start - purge_delta if (self.purged_weeks > 0 and not is_prev_adjacent) else test_start
            )
            purge_after_end = test_end + purge_delta if (self.purged_weeks > 0 and not is_next_adjacent) else test_end
            embargo_end = (
                purge_after_end + embargo_delta
                if (self.embargo_weeks > 0 and not is_next_adjacent)
                else purge_after_end
            )
            ex.append((purge_before_start, embargo_end))
        return self._merge_intervals(ex)

    def _optimal_folds_number(
        self,
        n_observations: int,
        target_train_size: int,
        target_n_test_paths: int,
    ) -> tuple[int, int]:
        def _n_splits(n_folds: int, n_test_folds: int) -> int:
            return math.comb(n_folds, n_test_folds)

        def _n_test_paths(n_folds: int, n_test_folds: int) -> int:
            return _n_splits(n_folds, n_test_folds) * n_test_folds // n_folds

        def _avg_train_size(n_obs: int, n_folds: int, n_test_folds: int) -> float:
            return n_obs / n_folds * (n_folds - n_test_folds)

        def _cost(n_folds: int, n_test_folds: int) -> float:
            n_paths = _n_test_paths(n_folds, n_test_folds)
            avg_train = _avg_train_size(n_observations, n_folds, n_test_folds)
            return self.weight_n_test_paths * abs(n_paths - target_n_test_paths) / max(
                target_n_test_paths, 1
            ) + self.weight_train_size * abs(avg_train - target_train_size) / max(target_train_size, 1)

        costs: list[float] = []
        candidates: list[tuple[int, int]] = []
        upper = min(n_observations, self.max_folds)
        for n_folds in range(3, max(4, upper + 1)):
            for n_test_folds in range(2, n_folds):
                candidates.append((n_folds, n_test_folds))
                costs.append(_cost(n_folds, n_test_folds))
        if not costs:
            return 3, 2
        optimal_idx = int(np.argmin(costs))
        return candidates[optimal_idx]

    def _calculate_optimal_params(self, df: pd.DataFrame) -> tuple[int, int]:
        n_observations = len(df)
        target_train_size = int(n_observations * self.target_train_size_pct)
        optimal_n_folds, optimal_n_test_folds = self._optimal_folds_number(
            n_observations=n_observations,
            target_train_size=target_train_size,
            target_n_test_paths=self.target_n_test_paths,
        )
        months_full = self._get_total_months(df)
        optimal_test_months = max(1, math.floor(months_full / optimal_n_folds))
        return optimal_test_months, optimal_n_test_folds

    def get_params(self, df: pd.DataFrame) -> dict[str, int]:
        if self.test_months is None or self.S is None:
            opt_test_months, opt_S = self._calculate_optimal_params(df)
            actual_test_months = self.test_months if self.test_months is not None else opt_test_months
            actual_S = self.S if self.S is not None else opt_S
        else:
            actual_test_months = self.test_months
            actual_S = self.S
        return {
            "test_months": actual_test_months,
            "S": actual_S,
            "purged_weeks": self.purged_weeks,
            "embargo_weeks": self.embargo_weeks,
            "total_months": self._get_total_months(df),
        }

    def split_paths(self, df: pd.DataFrame) -> dict[int, list[tuple[pd.DataFrame, pd.DataFrame]]]:
        if self.test_months is None or self.S is None:
            opt_test_months, opt_S = self._calculate_optimal_params(df)
            if self.test_months is None:
                self.test_months = opt_test_months
            if self.S is None:
                self.S = opt_S

        if df.empty:
            return {}

        df = self._sort_df(df)
        idx = self._resolve_index(df)
        group_starts, group_ends = self._build_month_groups(df)
        N = len(group_starts)
        if N < self.S:
            raise ValueError(f"Not enough month groups for S={self.S}; got N={N}.")

        # Safe path generation
        combs_gen = combinations(range(N), self.S)
        if self.max_paths_to_return is None:
            if self.randomly_sample_paths:
                raise ValueError("max_paths_to_return required for random sampling with unlimited paths")
            paths = list(combs_gen)
        else:
            desired = self.max_paths_to_return
            if self.randomly_sample_paths:
                if self.random_state is not None:
                    random.seed(self.random_state)
                paths = [tuple(sorted(random.sample(range(N), self.S))) for _ in range(desired)]
            else:
                paths = list(islice(combs_gen, desired))

        purge_delta = pd.Timedelta(weeks=self.purged_weeks)
        embargo_delta = pd.Timedelta(weeks=self.embargo_weeks)

        out: dict[int, list[tuple[pd.DataFrame, pd.DataFrame]]] = {}
        for path_id, comb in enumerate(paths):
            exclusions = self._build_exclusion_intervals_for_path(
                comb, group_starts, group_ends, purge_delta, embargo_delta
            )
            include_mask = self._mask_include_from_exclusions(idx, exclusions)
            train_df = df.loc[include_mask].copy()
            if train_df.empty:
                continue
            folds: list[tuple[pd.DataFrame, pd.DataFrame]] = []
            for g in comb:
                test_start, test_end = group_starts[g], group_ends[g]
                i = idx.searchsorted(test_start, side="left")
                j = idx.searchsorted(test_end, side="left")
                test_df = df.iloc[i:j].copy()
                if not test_df.empty:
                    folds.append((train_df, test_df))
            if folds:
                out[path_id] = folds
        return out

    def split_paths_with_holdout(
        self, df: pd.DataFrame
    ) -> tuple[dict[int, list[tuple[pd.DataFrame, pd.DataFrame]]], pd.DataFrame]:
        if self.holdout_pct <= 0.0 or self.holdout_pct >= 1.0:
            return self.split_paths(df), pd.DataFrame()
        df = self._sort_df(df)
        n = len(df)
        split_idx = int(n * (1.0 - self.holdout_pct))
        cv_df = df.iloc[:split_idx].copy()
        test_df = df.iloc[split_idx:].copy()
        cpcv_paths = self.split_paths(cv_df)
        return cpcv_paths, test_df

    def plot_paths(
        self,
        df: pd.DataFrame,
        paths: dict[int, list[tuple[pd.DataFrame, pd.DataFrame]]],
        max_paths_to_show: int = 10,
        style: PlotStyle = DEFAULT_STYLE,
    ) -> None:
        import plotly.graph_objects as go

        if not paths:
            raise ValueError("No CPCV paths to plot.")
        idx = self._resolve_index(df).sort_values()
        min_date, max_date = idx.min(), idx.max()
        purge_delta = pd.Timedelta(weeks=self.purged_weeks)
        embargo_delta = pd.Timedelta(weeks=self.embargo_weeks)
        fig = go.Figure()
        paths_to_show = min(len(paths), max_paths_to_show)
        for p in range(paths_to_show):
            fillcolor = "rgba(255,255,255,0.03)" if p % 2 == 0 else "rgba(0,0,0,0.00)"
            fig.add_shape(
                type="rect",
                xref="x",
                yref="y",
                x0=min_date,
                x1=max_date,
                y0=p - 0.45,
                y1=p + 0.45,
                line=dict(width=0),
                fillcolor=fillcolor,
                layer="below",
            )
        colors = {
            "Train": style.accent1,
            "Test": style.accent2,
            "Purge": style.accent6,
            "Embargo": style.accent3,
        }
        legend_added = set()

        def add_segment(y, x0, x1, color, name, dash=None, width=6):
            if x0 >= x1:
                return
            show = name not in legend_added
            if show:
                legend_added.add(name)
            fig.add_trace(
                go.Scatter(
                    x=[x0, x1],
                    y=[y, y],
                    mode="lines",
                    line=dict(color=color, width=width, dash=dash),
                    name=name if show else None,
                    legendgroup=name,
                    showlegend=show,
                    hovertemplate=f"<b>{name}</b><br>%{{x|%b %d, %Y %H:%M %Z}}<extra></extra>",
                    connectgaps=False,
                )
            )

        for p, (_path_id, folds) in enumerate(list(paths.items())[:max_paths_to_show]):
            test_periods = []
            for _, test_df in folds:
                test_idx = self._resolve_index(test_df)
                test_start = test_idx.min()
                test_end = test_idx.max()
                test_periods.append((test_start, test_end))
            test_periods = sorted(test_periods, key=lambda x: x[0])
            exclusions = []
            for i, (test_start, test_end) in enumerate(test_periods):
                is_prev_adjacent = i > 0 and test_start == test_periods[i - 1][1]
                is_next_adjacent = i < len(test_periods) - 1 and test_periods[i + 1][0] == test_end
                purge_before = (
                    test_start - purge_delta if (self.purged_weeks > 0 and not is_prev_adjacent) else test_start
                )
                purge_after = test_end + purge_delta if (self.purged_weeks > 0 and not is_next_adjacent) else test_end
                embargo_end = (
                    purge_after + embargo_delta if (self.embargo_weeks > 0 and not is_next_adjacent) else purge_after
                )
                exclusions.append((purge_before, embargo_end))
            exclusions = self._merge_intervals(exclusions)
            cursor = min_date
            allowed = []
            for start, end in exclusions:
                if start > cursor:
                    allowed.append((cursor, start))
                cursor = max(cursor, end)
            if cursor < max_date:
                allowed.append((cursor, max_date))
            for start, end in allowed:
                add_segment(p, start, end, colors["Train"], "Train", dash=None, width=6)
            for i, (test_start, test_end) in enumerate(test_periods):
                is_prev_adjacent = i > 0 and test_start == test_periods[i - 1][1]
                is_next_adjacent = i < len(test_periods) - 1 and test_periods[i + 1][0] == test_end
                if self.purged_weeks > 0 and not is_prev_adjacent:
                    purge_before = test_start - purge_delta
                    add_segment(p, purge_before, test_start, colors["Purge"], "Purge", dash="dash", width=4)
                add_segment(p, test_start, test_end, colors["Test"], "Test", dash=None, width=6)
                if self.purged_weeks > 0 and not is_next_adjacent:
                    purge_after = test_end + purge_delta
                    add_segment(p, test_end, purge_after, colors["Purge"], "Purge", dash="dash", width=4)
                if self.embargo_weeks > 0 and not is_next_adjacent:
                    embargo_end = purge_after + embargo_delta if self.purged_weeks > 0 else test_end + embargo_delta
                    add_segment(
                        p,
                        max(test_end, purge_after if self.purged_weeks > 0 else test_end),
                        embargo_end,
                        colors["Embargo"],
                        "Embargo",
                        dash="dot",
                        width=4,
                    )
        style.apply(fig)
        fig.update_layout(
            title="CPCV Paths Visualization",
            xaxis_title="Date",
            yaxis=dict(
                tickmode="array",
                tickvals=list(range(paths_to_show)),
                ticktext=[f"Path {i + 1}" for i in range(paths_to_show)],
                zeroline=False,
                showgrid=False,
            ),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
                bgcolor=style.plot_bgcolor,
                bordercolor=style.line,
                borderwidth=1,
            ),
            hovermode="x unified",
        )
        height = max(380, 28 * paths_to_show + 140)
        fig.update_layout(height=height)
        fig.show()

    def get_train_test_for_path(
        self, df: pd.DataFrame, paths: dict[int, list[tuple[pd.DataFrame, pd.DataFrame]]], path_id: int
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        if path_id not in paths:
            raise ValueError(f"Path ID {path_id} not found in paths dictionary.")
        folds = paths[path_id]
        test_dfs = [test_df for _, test_df in folds]
        full_test_df = self._sort_df(pd.concat(test_dfs))
        train_dfs = [train_df for train_df, _ in folds]
        full_train_df = pd.concat(train_dfs)
        full_train_df = full_train_df.drop_duplicates().pipe(self._sort_df)
        return full_train_df, full_test_df
