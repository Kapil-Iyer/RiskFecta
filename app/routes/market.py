"""GET /api/market/summary — truthful descriptive facts already known.

No analytics, forecasts, or derived metrics are invented here — only counts
and date bounds read straight from `prices_raw`.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends

from app.db import get_db
from app.schemas import MarketSummaryResponse

router = APIRouter(prefix="/api", tags=["market"])


@router.get("/market/summary", response_model=MarketSummaryResponse)
def market_summary(conn=Depends(get_db)) -> MarketSummaryResponse:
    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(DISTINCT ticker), COUNT(*), MIN(date), MAX(date) FROM prices_raw")
        ticker_count, row_count, first_date, last_date = cur.fetchone()

    return MarketSummaryResponse(
        ticker_count=ticker_count or 0,
        price_row_count=row_count or 0,
        first_date=first_date,
        last_date=last_date,
    )
