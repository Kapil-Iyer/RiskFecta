"""GET /api/predictions, GET /api/predictions/dates — real, frozen Phase 4-6
walk-forward forecast cross-sections.

These are historical out-of-sample research results (ML_SPEC.md §14, §19-20;
BUILD_PLAN.md Phase 4-6), never a live/current-day market forecast. This
router only ever SELECTs from `predictions` — it must never write to it
(TRD.md §9; the Phase 2A/8B read-only API contract).

Ranking by model (Ensemble/XGBoost/LSTM) is deliberately a client-side
concern, not a query parameter here: the cross-section for a formation date
is small (50 rows) and returning all three model columns in one response
lets the frontend re-rank on model switch without a repeat request, and
avoids ever mapping a query parameter onto a SQL column name.
"""
from __future__ import annotations

from datetime import date as date_type
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query

from app.db import get_db
from app.schemas import PredictionCrossSectionResponse, PredictionRow

router = APIRouter(prefix="/api", tags=["predictions"])


@router.get("/predictions/dates", response_model=List[date_type])
def get_prediction_dates(conn=Depends(get_db)) -> List[date_type]:
    """Every historical walk-forward formation date with persisted
    forecasts, ascending — the full frozen Phase 4-6 calendar (47 dates),
    not a live schedule."""
    with conn.cursor() as cur:
        cur.execute("SELECT DISTINCT forecast_date FROM predictions ORDER BY forecast_date ASC")
        return [r[0] for r in cur.fetchall()]


@router.get("/predictions", response_model=PredictionCrossSectionResponse)
def get_predictions(
    formation_date: Optional[date_type] = Query(
        None,
        description=(
            "Historical walk-forward formation date (YYYY-MM-DD). "
            "Defaults to the latest available formation date."
        ),
    ),
    conn=Depends(get_db),
) -> PredictionCrossSectionResponse:
    with conn.cursor() as cur:
        if formation_date is None:
            cur.execute("SELECT MAX(forecast_date) FROM predictions")
            row = cur.fetchone()
            formation_date = row[0] if row else None
            if formation_date is None:
                raise HTTPException(status_code=404, detail="No prediction data available")

        cur.execute(
            "SELECT ticker, target_date, xgb_pred, lstm_pred, ensemble_pred, "
            "actual_return, directional_correct "
            "FROM predictions WHERE forecast_date = %s ORDER BY ticker ASC",
            (formation_date,),
        )
        rows = cur.fetchall()

    if not rows:
        raise HTTPException(status_code=404, detail=f"No predictions for formation date {formation_date}")

    target_date = rows[0][1]
    predictions = [
        PredictionRow(
            ticker=r[0],
            xgb_pred=float(r[2]) if r[2] is not None else None,
            lstm_pred=float(r[3]) if r[3] is not None else None,
            ensemble_pred=float(r[4]) if r[4] is not None else None,
            actual_return=float(r[5]) if r[5] is not None else None,
            directional_correct=r[6],
        )
        for r in rows
    ]
    return PredictionCrossSectionResponse(
        formation_date=formation_date,
        target_date=target_date,
        count=len(predictions),
        predictions=predictions,
    )
