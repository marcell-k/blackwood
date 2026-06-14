from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd
from src.config import LIVE_START, TEST_END, TRAIN_END


@dataclass
class AllocatorConfig:
    cadence: str = "W-FRI"
    rolling_window: int = 63
    min_history: int = 63
    score_weight_sharpe: float = 0.5
    score_weight_drawdown: float = -0.3
    score_weight_hitrate: float = 0.2
    k: float = 0.35
    m_min: float = 0.5
    m_max: float = 1.5
    alpha: float = 0.7
    weight_floor: float = 0.0
    weight_cap: float = 1.0
    turnover_cap: float | None = None
    eps: float = 1e-12
    train_end: str = TRAIN_END
    test_end: str = TEST_END
    live_start: str = LIVE_START


class CentralRiskAllocator:
    """
    Central risk allocator for strategy-level overlays.

    The allocator is fit on train data only, then computes rolling health scores from
    realized vs expected metrics and maps them to multiplicative strategy budgets.
    """

    def __init__(self, config: AllocatorConfig | None = None):
        self.config = config or AllocatorConfig()
        self.config.min_history = max(self.config.min_history, self.config.rolling_window)

        self.expected_profile = pd.DataFrame()
        self.metric_scales = pd.DataFrame()
        self.current_multipliers = pd.Series(dtype=float)
        self.last_score_components: dict[str, dict[str, Any]] = {}
        self.last_overlay_stats: dict[str, Any] = {}
        self.last_compute_as_of: pd.Timestamp | None = None
        self.is_fitted: bool = False

    @staticmethod
    def _align_timestamp_to_index_tz(ts: pd.Timestamp, idx: pd.DatetimeIndex) -> pd.Timestamp:
        if idx.tz is None:
            return ts.tz_localize(None) if ts.tz is not None else ts
        if ts.tz is None:
            return ts.tz_localize(idx.tz)
        return ts.tz_convert(idx.tz)

    def _compute_window_metrics(self, returns: pd.Series) -> dict[str, float]:
        returns = returns.astype(float).dropna()
        if returns.empty:
            return {"sharpe": 0.0, "volatility": 0.0, "drawdown": 0.0, "hitrate": 0.0}

        ann_vol = float(returns.std(ddof=1) * np.sqrt(252))
        if not np.isfinite(ann_vol):
            ann_vol = 0.0

        if ann_vol > self.config.eps:
            ann_ret = float(returns.mean() * 252.0)
            sharpe = ann_ret / ann_vol
        else:
            sharpe = 0.0

        equity = (1.0 + returns).cumprod()
        running_max = equity.cummax()
        dd = (equity - running_max) / running_max
        drawdown = float(abs(dd.min())) if len(dd) > 0 else 0.0

        hitrate = float((returns > 0).mean())
        if not np.isfinite(hitrate):
            hitrate = 0.0

        return {
            "sharpe": float(sharpe),
            "volatility": float(max(ann_vol, 0.0)),
            "drawdown": float(max(drawdown, 0.0)),
            "hitrate": float(np.clip(hitrate, 0.0, 1.0)),
        }

    def _rolling_metric_history(self, returns: pd.Series) -> pd.DataFrame:
        returns = returns.astype(float).dropna()
        window = self.config.rolling_window
        if len(returns) < window:
            return pd.DataFrame(columns=["sharpe", "volatility", "drawdown", "hitrate"])

        rows = []
        idxs = []
        for end in range(window, len(returns) + 1):
            r = returns.iloc[end - window : end]
            rows.append(self._compute_window_metrics(r))
            idxs.append(r.index[-1])

        return pd.DataFrame(rows, index=idxs)

    def fit(self, train_returns: pd.DataFrame) -> None:
        if not isinstance(train_returns, pd.DataFrame):
            raise TypeError("train_returns must be a pandas DataFrame")
        if train_returns.empty:
            raise ValueError("train_returns is empty")

        required = max(self.config.min_history, self.config.rolling_window, 2)
        expected_rows: dict[str, dict[str, float]] = {}
        scale_rows: dict[str, dict[str, float]] = {}

        for strategy in train_returns.columns:
            series = train_returns[strategy].astype(float).dropna()
            if len(series) < required:
                continue

            expected_metrics = self._compute_window_metrics(series)
            rolling_metrics = self._rolling_metric_history(series)

            if rolling_metrics.empty:
                sharpe_scale = max(abs(expected_metrics["sharpe"]), 1.0)
                dd_scale = max(abs(expected_metrics["drawdown"]), 0.10)
                hitrate_scale = max(abs(expected_metrics["hitrate"]), 0.10)
            else:
                sharpe_scale = float(rolling_metrics["sharpe"].std(ddof=1))
                dd_scale = float(rolling_metrics["drawdown"].std(ddof=1))
                hitrate_scale = float(rolling_metrics["hitrate"].std(ddof=1))

                if not np.isfinite(sharpe_scale) or sharpe_scale <= self.config.eps:
                    sharpe_scale = max(abs(expected_metrics["sharpe"]), 1.0)
                if not np.isfinite(dd_scale) or dd_scale <= self.config.eps:
                    dd_scale = max(abs(expected_metrics["drawdown"]), 0.10)
                if not np.isfinite(hitrate_scale) or hitrate_scale <= self.config.eps:
                    hitrate_scale = max(abs(expected_metrics["hitrate"]), 0.10)

            expected_rows[strategy] = expected_metrics
            scale_rows[strategy] = {
                "sharpe": float(sharpe_scale),
                "drawdown": float(dd_scale),
                "hitrate": float(hitrate_scale),
            }

        if not expected_rows:
            raise ValueError(f"No strategies have sufficient train history. Need >= {required} points per strategy.")

        self.expected_profile = pd.DataFrame.from_dict(expected_rows, orient="index")
        self.metric_scales = pd.DataFrame.from_dict(scale_rows, orient="index")
        self.current_multipliers = pd.Series(1.0, index=self.expected_profile.index, dtype=float)
        self.last_score_components = {}
        self.last_overlay_stats = {}
        self.last_compute_as_of = None
        self.is_fitted = True

    def compute_multipliers(self, as_of: pd.Timestamp, returns_hist: pd.DataFrame) -> pd.Series:

        hist = returns_hist.copy()
        hist.index = pd.to_datetime(hist.index)

        as_of_ts = self._align_timestamp_to_index_tz(pd.Timestamp(as_of), pd.DatetimeIndex(hist.index))
        hist = hist.loc[hist.index < as_of_ts]

        multipliers = pd.Series(1.0, index=self.expected_profile.index, dtype=float)
        details: dict[str, dict[str, Any]] = {}

        for strategy in self.expected_profile.index:
            if strategy not in hist.columns:
                details[strategy] = {"status": "missing_series"}
                continue

            series = hist[strategy].astype(float).dropna()
            if len(series) < self.config.min_history:
                details[strategy] = {
                    "status": "cold_start",
                    "history_len": len(series),
                    "required_history": int(self.config.min_history),
                }
                multipliers[strategy] = 1.0
                continue

            window = series.iloc[-self.config.rolling_window :]
            realized = self._compute_window_metrics(window)
            expected = self.expected_profile.loc[strategy]
            scales = self.metric_scales.loc[strategy]

            z_sharpe = (realized["sharpe"] - float(expected["sharpe"])) / float(scales["sharpe"])
            z_dd = (realized["drawdown"] - float(expected["drawdown"])) / float(scales["drawdown"])
            z_hitrate = (realized["hitrate"] - float(expected["hitrate"])) / float(scales["hitrate"])

            score = (
                self.config.score_weight_sharpe * z_sharpe
                + self.config.score_weight_drawdown * z_dd
                + self.config.score_weight_hitrate * z_hitrate
            )

            m_raw = float(np.clip(np.exp(self.config.k * score), self.config.m_min, self.config.m_max))
            prev = float(self.current_multipliers.get(strategy, 1.0))
            m_smooth = self.config.alpha * prev + (1.0 - self.config.alpha) * m_raw
            m_smooth = float(np.clip(m_smooth, self.config.m_min, self.config.m_max))

            multipliers[strategy] = m_smooth
            details[strategy] = {
                "status": "active",
                "history_len": len(series),
                "z_sharpe": float(z_sharpe),
                "z_drawdown": float(z_dd),
                "z_hitrate": float(z_hitrate),
                "score": float(score),
                "m_raw": float(m_raw),
                "m_smooth": float(m_smooth),
                "realized": realized,
                "expected": {
                    "sharpe": float(expected["sharpe"]),
                    "volatility": float(expected["volatility"]),
                    "drawdown": float(expected["drawdown"]),
                    "hitrate": float(expected["hitrate"]),
                },
            }

        self.current_multipliers = multipliers.copy()
        self.last_score_components = details
        self.last_compute_as_of = as_of_ts
        return multipliers

    def _project_weights_to_bounds(self, weights: pd.Series) -> tuple[pd.Series, bool]:
        w = weights.astype(float).fillna(0.0).copy()
        if w.empty:
            return w, False

        n = len(w)
        floor = float(np.clip(self.config.weight_floor, 0.0, 1.0))
        cap = float(np.clip(self.config.weight_cap, floor, 1.0))

        if floor * n > 1.0:
            floor = 1.0 / n
        if cap * n < 1.0:
            cap = max(cap, 1.0 / n)

        total = float(w.sum())
        if total <= self.config.eps:
            w[:] = 1.0 / n
        else:
            w /= total

        original = w.copy()
        w = w.clip(lower=floor, upper=cap)

        for _ in range(4 * n + 5):
            s = float(w.sum())
            if abs(s - 1.0) <= 1e-10:
                break

            if s < 1.0:
                room = (cap - w).clip(lower=0.0)
                room_sum = float(room.sum())
                if room_sum <= self.config.eps:
                    break
                w += room * ((1.0 - s) / room_sum)
            else:
                slack = (w - floor).clip(lower=0.0)
                slack_sum = float(slack.sum())
                if slack_sum <= self.config.eps:
                    break
                w -= slack * ((s - 1.0) / slack_sum)

            w = w.clip(lower=floor, upper=cap)

        final_sum = float(w.sum())
        if final_sum <= self.config.eps:
            w[:] = 1.0 / n
        elif abs(final_sum - 1.0) > 1e-10:
            w /= final_sum

        clipped = not np.allclose(w.values, original.values, atol=1e-10)
        return w, clipped

    def apply_overlay(self, base_weights: pd.Series, multipliers: pd.Series) -> pd.Series:
        if not isinstance(base_weights, pd.Series):
            raise TypeError("base_weights must be a pandas Series")
        if base_weights.empty:
            raise ValueError("base_weights is empty")

        base = base_weights.astype(float).fillna(0.0).copy()
        base_sum = float(base.sum())
        if base_sum <= self.config.eps:
            base[:] = 1.0 / len(base)
        else:
            base /= base_sum

        mult = multipliers.reindex(base.index).astype(float).fillna(1.0)
        adjusted = (base * mult).astype(float)
        adjusted_sum = float(adjusted.sum())
        if adjusted_sum <= self.config.eps:
            adjusted = base.copy()
        else:
            adjusted /= adjusted_sum

        bounded, bounds_clipped = self._project_weights_to_bounds(adjusted)
        turnover = float(np.abs(bounded - base).sum())
        turnover_limit_applied = False
        turnover_cap = self.config.turnover_cap

        if turnover_cap is not None and turnover_cap > 0 and turnover > turnover_cap + self.config.eps:
            lam = float(turnover_cap / turnover)
            candidate = base + lam * (bounded - base)
            candidate, candidate_clipped = self._project_weights_to_bounds(candidate)
            candidate_turnover = float(np.abs(candidate - base).sum())

            if candidate_turnover <= turnover_cap + 1e-9:
                bounded = candidate
                turnover = candidate_turnover
                turnover_limit_applied = True
                bounds_clipped = bounds_clipped or candidate_clipped

        self.last_overlay_stats = {
            "turnover": float(turnover),
            "turnover_cap": float(turnover_cap) if turnover_cap is not None else None,
            "turnover_limit_applied": bool(turnover_limit_applied),
            "bounds_clipped": bool(bounds_clipped),
            "base_weight_sum": float(base.sum()),
            "adjusted_weight_sum": float(bounded.sum()),
        }
        return bounded

    def get_state_snapshot(self) -> dict[str, Any]:
        return {
            "is_fitted": bool(self.is_fitted),
            "config": asdict(self.config),
            "last_compute_as_of": self.last_compute_as_of,
            "expected_profile": self.expected_profile.to_dict(orient="index"),
            "metric_scales": self.metric_scales.to_dict(orient="index"),
            "current_multipliers": self.current_multipliers.to_dict(),
            "last_score_components": self.last_score_components,
            "last_overlay_stats": self.last_overlay_stats,
        }
