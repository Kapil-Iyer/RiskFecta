"""
RiskFecta configuration — V2 baseline (horizon/universe locked to Bloomberg export).
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
# Rolling-window constants (locked V2 — trading SESSIONS, not calendar rows)
# ---------------------------------------------------------------------------
TRAIN_WINDOW = 252   # trading sessions (~1 year)
STEP = 21            # trading sessions (~1 month rebalance)
LSTM_SEQ = 60        # input sequence length (trading sessions)
FORECAST_HORIZON = 21  # 21-session forward return target

# ---------------------------------------------------------------------------
# Momentum lookbacks (Phase 3 decision gate — LOCKED)
# ML_SPEC.md §6/§15 name "momentum (3-month, 6-month equivalent in trading
# sessions)" without freezing an exact session count. This project adopts
# the standard ~21-trading-sessions-per-month approximation (3m -> 63,
# 6m -> 126) as the RiskFecta V2 Phase 3 methodology, locked here as the
# single source of truth — never hardcoded independently inside
# pipeline/features.py. Fixed for all later phases unless a future,
# explicitly approved methodology revision changes it.
# ---------------------------------------------------------------------------
MOMENTUM_3M_SESSIONS = 63
MOMENTUM_6M_SESSIONS = 126

# ---------------------------------------------------------------------------
# Ticker universe: exact 50 symbols from Bloomberg data/raw export
# (25 Information Technology + 25 Financials). Source of truth = CSVs, not hand list.
# ---------------------------------------------------------------------------
TICKER_UNIVERSE = [
    # Information Technology (Bloomberg export order)
    "AAPL", "MSFT", "NVDA", "AVGO", "ORCL", "CRM", "ADBE", "AMD", "QCOM", "TXN",
    "INTC", "MU", "AMAT", "KLAC", "SNPS", "CDNS", "PANW", "NOW", "APH", "MSI",
    "ADI", "MRVL", "IBM", "HPQ", "GLW",
    # Financials (Bloomberg export order; MRSH as exported — not MMC)
    "JPM", "BAC", "WFC", "GS", "MS", "C", "BLK", "SCHW", "AXP", "USB",
    "PNC", "TFC", "COF", "MRSH", "ICE", "CME", "SPGI", "MCO", "MSCI", "CB",
    "PGR", "MET", "AIG", "TRV", "AJG",
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
