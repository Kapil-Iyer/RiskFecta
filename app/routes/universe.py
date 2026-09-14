"""GET /api/universe — the real 50-stock universe (TRD.md §9; PRD.md §6)."""
from __future__ import annotations

import logging
from typing import Dict, List, Optional

import pandas as pd
from fastapi import APIRouter, Depends

from app.db import get_db
from app.schemas import UniverseTicker

router = APIRouter(prefix="/api", tags=["universe"])

logger = logging.getLogger(__name__)


def _load_sector_by_ticker() -> Dict[str, Optional[str]]:
    """Best-effort, read-only sector lookup from the already-validated static
    snapshot (`pipeline.normalize.normalize_static_fields`).

    Single-snapshot metadata only (ML_SPEC.md §2/§10) — never attached to
    historical price rows, never persisted to a new database table. Returns
    {} if the gitignored source CSV isn't present locally (e.g. in CI)
    rather than failing the /api/universe request.
    """
    try:
        from pipeline.normalize import normalize_static_fields

        static_df = normalize_static_fields()
    except Exception:
        logger.info("Static snapshot unavailable; serving /api/universe without sector metadata.")
        return {}

    return {
        row.ticker: (row.sector if pd.notna(row.sector) else None)
        for row in static_df.itertuples(index=False)
    }


@router.get("/universe", response_model=List[UniverseTicker])
def get_universe(conn=Depends(get_db)) -> List[UniverseTicker]:
    """Real, stored tickers from `prices_raw` (not a hand-typed list),
    optionally enriched with read-only sector metadata."""
    with conn.cursor() as cur:
        cur.execute("SELECT DISTINCT ticker FROM prices_raw ORDER BY ticker")
        tickers = [r[0] for r in cur.fetchall()]

    sector_by_ticker = _load_sector_by_ticker()
    return [UniverseTicker(ticker=t, sector=sector_by_ticker.get(t)) for t in tickers]
