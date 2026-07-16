import os
from pathlib import Path

IS_MONTHS: int = 10
OOS_MONTHS: int = 3
KELLY_FRACTION: float = 0.1
CASH: float = 10e6
MARGIN: float = 1 / 100
TIMEZONE: str = "America/New_York"
RANDOM_STATE: int = 89
ANNUAL_TRADING_DAYS: int = 252
RISK_FREE_RATE: float = 0.0
SPLIT_TIME: str = "2024.01.01"
TRAIN_END: str = "2023-12-31"
TEST_END: str = "2025-11-30"
LIVE_START: str = "2025-12-01"


DATA_DIR = Path(os.environ.get("BLACKWOOD_DATA_PATH", "."))
NEWS_PATH = Path(os.environ.get("BLACKWOOD_NEWS_PATH", "."))
