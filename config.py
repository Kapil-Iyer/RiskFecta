"""
RiskFecta configuration — PRD v1.0 / Build Plan v1.0.
Ticker universe, feature lists, rolling-window constants, paths.
"""

import os
from pathlib import Path

# Load .env from project root so DATABASE_URL is available without manual export
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent / ".env")
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Paths (data/raw/ gitignored; use env or default)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"
MODELS_DIR = PROJECT_ROOT / "models"

# ---------------------------------------------------------------------------
# Rolling-window constants (locked — do not change without PRD amendment)
# ---------------------------------------------------------------------------
TRAIN_WINDOW = 252   # trading days (~1 year)
STEP = 21            # trading days (~1 month)
LSTM_SEQ = 60        # input sequence length (days)
FORECAST_HORIZON = 30  # 30-day forward return target

# ---------------------------------------------------------------------------
# Ticker universe: 40–50 S&P 500, Technology + Financial Services (placeholder).
# Replace with actual Bloomberg tickers after Phase 1 pull if needed.
# ---------------------------------------------------------------------------
TICKER_UNIVERSE = [
    # Technology (representative)
    "AAPL", "MSFT", "GOOGL", "GOOG", "META", "NVDA", "AVGO", "ORCL", "ADBE", "CRM",
    "CSCO", "ACN", "AMD", "INTC", "IBM", "QCOM", "TXN", "NOW", "INTU", "AMAT",
    "MU", "LRCX", "KLAC", "SNPS", "CDNS", "ADSK", "PANW", "CRWD", "FTNT", "WDAY",
    # Financial Services (representative)
    "JPM", "BAC", "WFC", "GS", "MS", "C", "BLK", "SCHW", "AXP", "SPGI",
    "MMC", "CB", "PGR", "MET", "AON", "ICE", "CME", "CBOE", "FIS", "FISV",
]

# ---------------------------------------------------------------------------
# Feature column names (must match schema and pipeline)
# ---------------------------------------------------------------------------
# LSTM: OHLCV + technical indicators (from features table / computed in features.py)
LSTM_PRICE_COLS = ["open", "high", "low", "close", "volume"]
LSTM_FEATURE_COLS = [
    "rsi_14", "macd", "macd_signal", "bb_upper", "bb_lower",
    "volatility_20d", "momentum_3m", "momentum_6m",
]
LSTM_ALL_FEATURES = LSTM_PRICE_COLS + LSTM_FEATURE_COLS

# XGBoost: tabular features per (ticker, date)
XGBOOST_FEATURE_COLS = [
    "beta", "mkt_cap_log", "div_yield", "sector",  # sector encoded in pipeline
    "vix", "yield_10y", "momentum_3m", "momentum_6m", "volatility_20d",
]

# ---------------------------------------------------------------------------
# Database (read from env; never hardcode credentials)
# ---------------------------------------------------------------------------
def get_database_url() -> str:
    url = os.environ.get("DATABASE_URL", "")
    if not url:
        raise ValueError("DATABASE_URL environment variable is not set (.env or export)")
    return url
