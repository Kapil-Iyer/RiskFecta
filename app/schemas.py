"""
Pydantic request/response schemas for the RiskFecta API (Phase 2A).

Every field here reflects data that already exists in `prices_raw` or the
Phase 1 static snapshot / project config — nothing here represents a
forecast, portfolio, or risk metric (those tables stay empty in Phase 2A).
"""
from __future__ import annotations

from datetime import date
from typing import List, Optional

from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str


class ReadinessResponse(BaseModel):
    status: str
    database: str


class UniverseTicker(BaseModel):
    ticker: str
    # Single-snapshot metadata only (pipeline.normalize.normalize_static_fields),
    # never a historical/predictive field. None if the local static snapshot
    # CSV isn't available (e.g. in CI, where data/raw/ is gitignored).
    sector: Optional[str] = None


class PriceObservation(BaseModel):
    date: date
    open: Optional[float] = None
    high: Optional[float] = None
    low: Optional[float] = None
    close: float
    volume: Optional[int] = None
    total_return_idx: Optional[float] = None


class PriceHistoryResponse(BaseModel):
    ticker: str
    start: Optional[date] = None
    end: Optional[date] = None
    count: int
    prices: List[PriceObservation]


class MarketSummaryResponse(BaseModel):
    ticker_count: int
    price_row_count: int
    first_date: Optional[date] = None
    last_date: Optional[date] = None
