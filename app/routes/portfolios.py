"""GET /api/portfolios, /api/portfolios/dates, /api/portfolios/strategies —
frozen, official Phase 7 historical portfolio-construction research
(ML_SPEC.md §23-27; BUILD_PLAN.md Phase 7).

Read-only, and reads ONLY the official `p7bv1_*` experiment rows — the
single frozen, already-verified run. Every number here is either:

  - a persisted official row (`portfolios.weight`/`target_return`/
    `portfolio_vol`/`sharpe_ratio`; `risk_metrics.realized_return_21`/
    `max_weight_observed`/`turnover`), or
  - a pure derivation from those same weights (active-holdings count,
    largest weight, concentration HHI = sum(w_i^2)).

Construction-time figures (`target_return`/`portfolio_vol`/`sharpe_ratio`)
are already persisted per-row and are READ DIRECTLY here — never
recomputed from mu/Sigma/rf. Reconstructing them would risk silently
drifting from the frozen Phase 7 result; reading them cannot.

Formation-date/strategy identity reuses the exact persistence-layer
convention (`optimizer.persistence.build_run_id`/`VALID_STRATEGIES`) rather
than inventing a second identity scheme. A `formation_date` that was never
Phase 7 portfolio-eligible (e.g. 2022-02-25, prediction-only) simply yields
no matching `run_id` and a clean 404 — no special-casing needed.
"""
from __future__ import annotations

from datetime import date as date_type
from typing import Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query

import config
from app.db import get_db
from app.schemas import (
    HoldingRow,
    PortfolioConstructionMetrics,
    PortfolioEvaluationMetrics,
    PortfolioResponse,
    StrategyInfo,
)
from optimizer.persistence import VALID_STRATEGIES, build_run_id, parse_run_id

router = APIRouter(prefix="/api", tags=["portfolios"])

EXPERIMENT_ID = "p7bv1"
DEFAULT_STRATEGY = "LW_MAXSHARPE"  # one of the frozen official strategies — not "the best one" (see StrategyInfo docs)

STRATEGY_REGISTRY: Dict[str, StrategyInfo] = {
    "SAMPLE_MINVOL": StrategyInfo(
        key="SAMPLE_MINVOL", label="Sample Min-Vol", covariance_estimator="Sample",
        objective="Minimum volatility — does not use expected returns", is_optimized=True,
    ),
    "SAMPLE_MAXSHARPE": StrategyInfo(
        key="SAMPLE_MAXSHARPE", label="Sample Max-Sharpe", covariance_estimator="Sample",
        objective="Maximum Sharpe ratio, using the frozen 50/50 ensemble expected returns", is_optimized=True,
    ),
    "LW_MINVOL": StrategyInfo(
        key="LW_MINVOL", label="Ledoit-Wolf Min-Vol", covariance_estimator="Ledoit-Wolf",
        objective="Minimum volatility — does not use expected returns", is_optimized=True,
    ),
    "LW_MAXSHARPE": StrategyInfo(
        key="LW_MAXSHARPE", label="Ledoit-Wolf Max-Sharpe", covariance_estimator="Ledoit-Wolf",
        objective="Maximum Sharpe ratio, using the frozen 50/50 ensemble expected returns", is_optimized=True,
    ),
    "EQUAL_WEIGHT": StrategyInfo(
        key="EQUAL_WEIGHT", label="Equal Weight", covariance_estimator="Not applicable",
        objective="Equal-weight benchmark — not an optimized portfolio", is_optimized=False,
    ),
}


def _official_formation_dates(conn) -> List[date_type]:
    """The real, persisted Phase 7 portfolio calendar — derived from every
    distinct official run_id, never independently regenerated from the
    covariance-eligibility rule (that rule already decided what got
    persisted; this only reads the result)."""
    with conn.cursor() as cur:
        cur.execute("SELECT DISTINCT run_id FROM portfolios WHERE run_id LIKE %s", (f"{EXPERIMENT_ID}\\_%",))
        run_ids = [r[0] for r in cur.fetchall()]
    dates = {date_type.fromisoformat(parse_run_id(rid)["formation_date"]) for rid in run_ids}
    return sorted(dates)


@router.get("/portfolios/dates", response_model=List[date_type])
def get_portfolio_dates(conn=Depends(get_db)) -> List[date_type]:
    return _official_formation_dates(conn)


@router.get("/portfolios/strategies", response_model=List[StrategyInfo])
def get_portfolio_strategies() -> List[StrategyInfo]:
    return [STRATEGY_REGISTRY[s] for s in VALID_STRATEGIES]


@router.get("/portfolios", response_model=PortfolioResponse)
def get_portfolio(
    formation_date: Optional[date_type] = Query(
        None,
        description="Phase 7 portfolio formation date (46 available, 2022-03-28 to 2026-01-02). Defaults to the latest.",
    ),
    strategy: str = Query(
        DEFAULT_STRATEGY,
        description="One of: " + ", ".join(VALID_STRATEGIES),
    ),
    conn=Depends(get_db),
) -> PortfolioResponse:
    if strategy not in VALID_STRATEGIES:
        raise HTTPException(status_code=400, detail=f"Unknown strategy {strategy!r}; expected one of {VALID_STRATEGIES}")

    if formation_date is None:
        dates = _official_formation_dates(conn)
        if not dates:
            raise HTTPException(status_code=404, detail="No official Phase 7 portfolio data available")
        formation_date = dates[-1]

    run_id = build_run_id(EXPERIMENT_ID, strategy, formation_date)

    with conn.cursor() as cur:
        cur.execute(
            "SELECT ticker, weight, target_return, portfolio_vol, sharpe_ratio "
            "FROM portfolios WHERE run_id = %s ORDER BY ticker ASC",
            (run_id,),
        )
        rows = cur.fetchall()

    if not rows:
        raise HTTPException(
            status_code=404,
            detail=f"No official Phase 7 portfolio for formation_date={formation_date} strategy={strategy}",
        )
    if len(rows) != len(config.TICKER_UNIVERSE):
        raise HTTPException(
            status_code=500,
            detail=f"Official portfolio {run_id!r} has {len(rows)} holdings, expected {len(config.TICKER_UNIVERSE)}",
        )

    holdings = [HoldingRow(ticker=r[0], weight=float(r[1])) for r in rows]
    weights = [h.weight for h in holdings]
    target_return, portfolio_vol, sharpe_ratio = rows[0][2], rows[0][3], rows[0][4]

    with conn.cursor() as cur:
        cur.execute("SELECT metric_name, metric_value FROM risk_metrics WHERE run_id = %s", (run_id,))
        metric_rows = {name: float(value) for name, value in cur.fetchall()}

    if "realized_return_21" not in metric_rows or "max_weight_observed" not in metric_rows:
        raise HTTPException(status_code=500, detail=f"Official portfolio {run_id!r} is missing required risk metrics")

    strategy_info = STRATEGY_REGISTRY[strategy]

    return PortfolioResponse(
        formation_date=formation_date,
        strategy=strategy_info,
        max_weight_constraint=config.MAX_WEIGHT if strategy_info.is_optimized else None,
        holdings=holdings,
        active_holdings_count=sum(1 for w in weights if w > 1e-9),
        largest_weight=max(weights),
        concentration_hhi=sum(w * w for w in weights),
        construction=PortfolioConstructionMetrics(
            expected_return_21=float(target_return) if target_return is not None else None,
            predicted_volatility_21=float(portfolio_vol) if portfolio_vol is not None else None,
            expected_sharpe_21=float(sharpe_ratio) if sharpe_ratio is not None else None,
        ),
        evaluation=PortfolioEvaluationMetrics(
            realized_return_21=metric_rows["realized_return_21"],
            turnover=metric_rows.get("turnover"),
            max_weight_observed=metric_rows["max_weight_observed"],
        ),
        source="official_phase7_persisted_experiment",
    )
