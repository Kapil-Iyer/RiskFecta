"""
Pydantic request/response schemas for the RiskFecta API.

Every field here reflects data that already exists in `prices_raw`, the
Phase 1 static snapshot, project config, or (as of Phase 8B) the frozen
Phase 4-6 `predictions` table. Schemas for the Phase 7 optimizer output are
not yet defined here — see BUILD_PLAN.md Phase 8C/8D.
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


class PredictionRow(BaseModel):
    """One ticker's forecast cross-section at a formation date (Phase 4-6,
    frozen historical walk-forward — never a live/current forecast).

    `actual_return`/`directional_correct` are ex-post evaluation fields
    filled in only after the 21-session target_date has passed — they were
    NOT available at formation time and must never be presented as if they
    were an input to the forecast itself.
    """

    ticker: str
    xgb_pred: Optional[float] = None
    lstm_pred: Optional[float] = None
    ensemble_pred: Optional[float] = None
    actual_return: Optional[float] = None
    directional_correct: Optional[bool] = None


class PredictionCrossSectionResponse(BaseModel):
    formation_date: date
    target_date: date
    count: int
    # Base order is deterministic (ticker ascending) — see
    # app/routes/predictions.py. Ranking by a chosen model is a client-side
    # concern, not an API parameter.
    predictions: List[PredictionRow]
