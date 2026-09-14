"""GET /api/prices/{ticker} — real historical observations from `prices_raw`."""
from __future__ import annotations

from datetime import date as date_type
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query

import config
from app.db import get_db
from app.schemas import PriceHistoryResponse, PriceObservation

router = APIRouter(prefix="/api", tags=["prices"])

# Canonical-symbol lookup, case-insensitive (config.TICKER_UNIVERSE is already
# the frozen, canonical 50-ticker universe — TRD.md/config.py).
_CANONICAL = {t.upper(): t for t in config.TICKER_UNIVERSE}


@router.get("/prices/{ticker}", response_model=PriceHistoryResponse)
def get_prices(
    ticker: str,
    start: Optional[date_type] = Query(None, description="Inclusive start date (YYYY-MM-DD)"),
    end: Optional[date_type] = Query(None, description="Inclusive end date (YYYY-MM-DD)"),
    conn=Depends(get_db),
) -> PriceHistoryResponse:
    canonical = _CANONICAL.get(ticker.upper())
    if canonical is None:
        raise HTTPException(status_code=404, detail=f"Unknown ticker: {ticker}")

    if start is not None and end is not None and start > end:
        raise HTTPException(status_code=400, detail="start date must not be after end date")

    sql = (
        "SELECT date, open, high, low, close, volume, total_return_idx "
        "FROM prices_raw WHERE ticker = %s"
    )
    params: List[object] = [canonical]
    if start is not None:
        sql += " AND date >= %s"
        params.append(start)
    if end is not None:
        sql += " AND date <= %s"
        params.append(end)
    sql += " ORDER BY date ASC"

    with conn.cursor() as cur:
        cur.execute(sql, tuple(params))
        rows = cur.fetchall()

    prices = [
        PriceObservation(
            date=r[0],
            open=float(r[1]) if r[1] is not None else None,
            high=float(r[2]) if r[2] is not None else None,
            low=float(r[3]) if r[3] is not None else None,
            close=float(r[4]),
            volume=int(r[5]) if r[5] is not None else None,
            total_return_idx=float(r[6]) if r[6] is not None else None,
        )
        for r in rows
    ]
    return PriceHistoryResponse(ticker=canonical, start=start, end=end, count=len(prices), prices=prices)
