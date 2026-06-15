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


DATA_DIR = Path(os.environ.get("BLACKWOOD_DATA_PATH"))
NEWS_PATH = Path(os.environ.get("BLACKWOOD_NEWS_PATH"))

BROKER_SPREADS = {
    # ===== Crypto =====
    "BTCUSD": 6.91834052072043e-05,  # 1200 spread --> 12 point
    "ETHUSD": 0.0005113466048348703,  # 290 spread --> 2.9 point
    # 'BTCEUR' : 0.0001, # 1200 spread --> 12 point
    # ===== Indices =====
    # 'CA60' : 0.0002195763274761348, # 80 spread --> 0.8 point
    "DAX": 1.0766209605181561e-05,  # 50 spread --> 0.5 point
    "US30": 1.2960362890160924e-05,  # 120 spread --> 1.2 point
    "US100": 4.0598255739703962e-05,  # 100 spread --> 1 point
    "US500": 3.780489649019341e-05,  # 50 spread --> 0.5 point
    "US2000": 5.059725847187846e-05,  # 35 spread --> 0.35 point
    "Europe50": 8.913291500285225e-05,  # 100 spread --> 1 point
    "France40": 5.048427968783476e-05,  # 120 spread --> 1.2 point
    "UK100": 4.701015419330576e-05,  # 90 spread --> 0.9 point (or 250 before session)
    "Spain35": 9.256288105052698e-05,  # 300 spread --> 3.0 point
    "Italy40": 0.0001035968146419778,  # 1000 spread --> 10 point
    "Netherlands25": 0.00010613593040879315,  # 20 spread --> 0.2 point
    "QQQ": 2.4928123909394578e-05,
    # 'SE30' : 7.947172973832127e-05, # 44 spread --> 0.44 point
    "HK50": 0.00017372505115237618,  # 900 spread --> 9 point (or 700 in session)
    "Japan225": 4.9809363626405384e-05,  # 200 spread --> 2 point (after session 4 point)
    "Switzerland20": 3.127859058670817e-05,  # 80 spread --> 0.8 point (1.7 point after session)
    # ===== FX =====
    "EURUSD": 4.336626278220595e-06,  # 0.1 pip half-spread
    "GBPUSD": 3.792446962629227e-06,  # 0.25 pip half-spread, total 0.5 pip
    "USDJPY": 3.1981374047754613e-06,  # 0.55 pip half-spread, total 1.1 pip
    "EURJPY": 1.3888040175322631e-05,  # 0.60 pip half-spread, total 1.2 pip
    "GBPJPY": 9.765005151040226e-06,  # 7 spread --> 0.07 point (can be 0.12 point)
    "EURCHF": 1.6095974933188018e-05,
    "USDCHF": 1.2402946940205394e-05,  # idk
    "EURGBP": 5.7264585290248485e-06,  # 1 but mostly 0
    "USDCAD": 3.5778687350958395e-06,  # 1 but mostly 0
    "AUDUSD": 0,
    # ===== Metals =====
    "XAUUSD": 1.2299336327811752e-05,  # 10 spread --> 0.1 point
    "XAGUSD": 0.00019515656879267687,
    # ===== Energy =====
    "NaturalGas": 0.002056597564988483,
    "USBrentCrudeOil": 8.026712900532974e-05,
    "USLightCrudeOil": 8.026712900532974e-05,
}
BROKER_COMMISSION = {
    # ===== Crypto =====
    "BTCUSD": (0, 0),
    "ETHUSD": (0, 0),
    # 'BTCEUR' : (3.5/100_000, 0),
    # ===== Indices =====
    "CA60": (0, 0),
    "DAX": (0, 0),
    "US30": (0, 0),
    "US100": (0, 0),
    "US500": (0, 0),
    "US2000": (0, 0),
    "Europe50": (0, 0),
    "France40": (0, 0),
    "UK100": (0, 0),
    "Spain35": (0, 0),
    "Italy40": (0, 0),
    "Netherlands25": (0, 0),
    "SE30": (0, 0),
    "Switzerland20": (0, 0),
    "HK50": (0, 0),
    "Japan225": (0, 0),
    "QQQ": (3.5 / 100_000, 0),
    # ===== FX =====
    "EURUSD": (3.5 / 100_000, 0),
    "EURCHF": (3.5 / 100_000, 0),
    "AUDUSD": (3.5 / 100_000, 0),
    "GBPUSD": (3.5 / 100_000, 0),
    "USDJPY": (3.5 / 100_000, 0),
    "EURJPY": (3.5 / 100_000, 0),
    "GBPJPY": (3.5 / 100_000, 0),
    "USDCHF": (3.5 / 100_000, 0),
    "EURGBP": (3.5 / 100_000, 0),
    "USDCAD": (3.5 / 100_000, 0),
    # ===== Metals =====
    "XAUUSD": (0, 0),
    "XAGUSD": (0, 0),
    # ===== Energy =====
    "NaturalGas": (0, 0),
    "USBrentCrudeOil": (0, 0),
    "USLightCrudeOil": (0, 0),
}
TIMEZONE_INSTRUMENT: dict[str, str] = {
    # ===== Indices (Americas) =====
    "US500": "US/Eastern",
    "US30": "US/Eastern",
    "US100": "US/Eastern",
    "US2000": "US/Eastern",
    "CA60": "America/Toronto",
    # ===== Indices (Europe) =====
    "DAX": "Europe/Berlin",
    "UK100": "Europe/Berlin",  # "Europe/London"
    "Spain35": "Europe/Berlin",  # "Europe/Madrid"
    "Italy40": "Europe/Berlin",  # "Europe/Rome"
    "France40": "Europe/Berlin",  # "Europe/Paris"
    "Europe50": "Europe/Berlin",
    "Switzerland20": "Europe/Berlin",  # "Europe/Zurich"
    "Netherlands25": "Europe/Berlin",  # "Europe/Amsterdam"
    "SE30": "Europe/Berlin",  # "Europe/Stockholm"
    # ===== Indices (Asia) =====
    "HK50": "Asia/Hong_Kong",
    "Japan225": "Asia/Tokyo",
    # ===== Metals =====
    "XAUUSD": "Europe/Berlin",
    "XAGUSD": "Europe/Berlin",
    # ===== Crypto =====
    "BTCUSD": "Europe/Berlin",
    "BTCEUR": "Europe/Berlin",
    "ETHUSD": "Europe/Berlin",
    # ===== FX =====
    "EURJPY": "Europe/Berlin",
    "EURCHF": "Europe/Berlin",
    "AUDUSD": "Europe/Berlin",
    "GBPJPY": "Europe/Berlin",
    "GBPUSD": "Europe/Berlin",
    "USDJPY": "Europe/Berlin",
    "EURUSD": "Europe/Berlin",
    "EURGBP": "Europe/Berlin",
    "USDCHF": "Europe/Berlin",
    "USDCAD": "Europe/Berlin",
    # ===== Energy =====
    "NaturalGas": "Europe/Berlin",
    "USBrentCrudeOil": "Europe/Berlin",
    "USLightCrudeOil": "Europe/Berlin",
}
US_OFFSET_INSTRUMENTS = {"US2000", "US100", "US30", "US500", "QQQ"}
ASSET_CLASSES = {
    "Crypto": ["BTCUSD", "ETHUSD", "BTCEUR"],
    "Indices All": [
        "CA60",
        "DAX",
        "US30",
        "US100",
        "US500",
        "US2000",
        "Europe50",
        "France40",
        "UK100",
        "Spain35",
        "Italy40",
        "Netherlands25",
        "SE30",
        "HK50",
        "Japan225",
        "Switzerland20",
        "QQQ",
    ],
    "Indices Eu": [
        "DAX",
        "Europe50",
        "France40",
        "UK100",
        "Spain35",
        "Italy40",
        "Netherlands25",
        "SE30",
        "",
    ],
    "Indices": ["DAX", "US30", "US100", "US500", "US2000", "UK100", "QQQ"],
    "Forex": [
        "EURUSD",
        "GBPUSD",
        "USDJPY",
        "EURJPY",
        "GBPJPY",
        "EURCHF",
        "USDCHF",
        "EURGBP",
        "USDCAD",
        "AUDUSD",
    ],
    "Metals": ["XAUUSD", "XAGUSD"],
    "Energy": ["USBrentCrudeOil", "USLightCrudeOil"],  # 'NaturalGas'
}


NEWS_CURRENCIES = {
    # ===== Crypto =====
    "BTCUSD": ("USD", ""),
    "ETHUSD": ("USD", ""),
    # 'BTCEUR' : ('USD', ''),
    # ===== Indices =====
    "CA60": ("USD", "CAD"),
    "DAX": ("USD", "EUR"),
    "US30": ("USD", ""),
    "US100": ("USD", ""),
    "US500": ("USD", ""),
    "US2000": ("USD", ""),
    "QQQ": ("USD", ""),
    "Europe50": ("USD", "EUR"),
    "France40": ("USD", "EUR"),
    "UK100": ("USD", "EUR"),
    "Spain35": ("USD", "EUR"),
    "Italy40": ("USD", "EUR"),
    "Netherlands25": ("USD", "EUR"),
    "SE30": ("USD", "EUR"),
    "Switzerland20": ("USD", "EUR"),
    "HK50": ("USD", "EUR"),
    "Japan225": ("USD", "JPN"),
    # ===== FX =====
    "EURUSD": ("EUR", "USD"),
    "EURCHF": ("EUR", "CHF"),
    "AUDUSD": ("AUD", "USD"),
    "GBPUSD": ("GBP", "USD"),
    "USDJPY": ("USD", "JPY"),
    "EURJPY": ("EUR", "JPY"),
    "GBPJPY": ("GBP", "JPY"),
    "USDCHF": ("USD", "CHF"),
    "EURGBP": ("EUR", "GBP"),
    "USDCAD": ("USD", "CAD"),
    # ===== Metals =====
    "XAUUSD": ("USD", ""),
    "XAGUSD": ("USD", ""),
    # ===== Energy =====
    "NaturalGas": ("USD", ""),
    "USBrentCrudeOil": ("USD", ""),
    "USLightCrudeOil": ("USD", ""),
}
