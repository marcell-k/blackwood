from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pandas as pd

type CPCVPaths = dict[int, list[tuple[pd.DataFrame, pd.DataFrame]]]
type ParameterSpace = dict[str, tuple[Any, ...] | list[int | float]]
type CommissionSpec = float | tuple[float, float] | Callable[..., float]
type ReturnsFrame = pd.DataFrame  # index=DatetimeIndex, columns=strategy/asset names
type StabilityMetrics = dict[str, dict[str, Any]]
