"""GET /api/risk — Risk Analytics: formation-time component risk
contribution to portfolio volatility for an official Phase 7 portfolio
(Phase 8D-1; PRD.md's Risk Analytics scope: "portfolio volatility,
concentration, per-asset risk contribution" — no VaR/CVaR/beta/factor
models/stress testing, none of which are frozen requirements for this
surface).

This describes the FROZEN Phase 7 portfolio at formation time — it never
constructs a new portfolio, never re-optimizes weights, and never touches
realized/ex-post data (no `realized_return_21`, no SPXT, no turnover).

RISK-CONTRIBUTION DEFINITION (locked for this slice — verified against
real official Phase 7 data before implementation; see the Phase 8D-1
report):

    sigma_p = sqrt(w^T Sigma_21 w)                    portfolio volatility
    MRC_i   = (Sigma_21 w)_i / sigma_p                marginal contribution
    RC_i    = w_i * MRC_i                             component contribution

`RC_i` is the AUTHORITATIVE decomposition (same units as `sigma_p`;
`sum_i RC_i == sigma_p` exactly, by the Euler identity for a
degree-1-homogeneous function). It is NEVER clamped, abs'd, or
renormalized — verified on real data that Equal Weight under Sample
covariance genuinely produces a small negative RC_i for at least one
name, and that must render as-is, not be hidden. `risk_share_i = RC_i /
sigma_p` is a convenience normalization for the UI only.

Covariance reconstruction reuses `app.routes.frontier.reconstruct_mu_sigma_rf`
verbatim (the SAME formation-time Sigma_21 the Efficient Frontier page
uses) — no second covariance implementation, no change to
`app/routes/frontier.py`, no risk to its own test suite.

Strategy -> covariance provenance is never ambiguous: for the four
optimized strategies, `covariance_estimator` is DETERMINED by the
strategy (read from the same `STRATEGY_REGISTRY` Portfolio Construction
uses) and a client-supplied `covariance` query param is rejected with 400
if it disagrees. EQUAL_WEIGHT has no covariance identity of its own (it
is never routed through Sample/Ledoit-Wolf optimization) — for it alone,
`covariance` selects which historical covariance estimate is used purely
to ANALYZE the fixed 1/50 benchmark's risk, and the response never
implies this was its construction methodology.

Read-only: no INSERT/UPDATE/DELETE anywhere in this module; nothing here
is persisted (risk contributions are deterministic derived analytics,
recomputed on every request).
"""
from __future__ import annotations

from datetime import date as date_type
from typing import List, Optional

import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Query

import config
from app.db import get_db
from app.routes.frontier import VALID_COVARIANCE_ESTIMATORS, reconstruct_mu_sigma_rf
from app.routes.portfolios import STRATEGY_REGISTRY, _official_formation_dates
from app.schemas import AssetRiskRow, PortfolioRiskSummary, RiskResponse, SectorRiskRow
from optimizer.persistence import VALID_STRATEGIES, build_run_id
from optimizer.portfolio import portfolio_volatility

router = APIRouter(prefix="/api", tags=["risk"])

EXPERIMENT_ID = "p7bv1"
TICKERS = list(config.TICKER_UNIVERSE)

# Same deterministic 25/25 split `frontend/src/data/tickerUniverse.ts`
# uses (Bloomberg export order: first 25 Information Technology, last 25
# Financials) — guaranteed available (unlike the gitignored static-fields
# snapshot `app/routes/universe.py` best-effort reads), so sector
# aggregation — a REQUIRED feature of this page, not decoration — never
# silently degrades to "sector unknown" in CI or a fresh checkout.
_IT_TICKERS = set(TICKERS[:25])
_SECTOR_IT = "Information Technology"
_SECTOR_FIN = "Financials"

_ESTIMATOR_LABEL_TO_KEY = {"Sample": "SAMPLE", "Ledoit-Wolf": "LW"}


def _sector_for_ticker(ticker: str) -> str:
    return _SECTOR_IT if ticker in _IT_TICKERS else _SECTOR_FIN


def _implied_covariance_key(strategy: str) -> Optional[str]:
    """The covariance estimator a strategy was actually CONSTRUCTED under,
    read from the same `STRATEGY_REGISTRY` Portfolio Construction uses —
    never a second, independently-hardcoded strategy->estimator map. Returns
    `None` for EQUAL_WEIGHT (no covariance identity of its own)."""
    label = STRATEGY_REGISTRY[strategy].covariance_estimator
    return _ESTIMATOR_LABEL_TO_KEY.get(label)


def _load_official_weights(conn, formation_date: date_type, strategy: str) -> np.ndarray:
    run_id = build_run_id(EXPERIMENT_ID, strategy, formation_date)
    with conn.cursor() as cur:
        cur.execute(
            "SELECT ticker, weight FROM portfolios WHERE run_id = %s ORDER BY ticker",
            (run_id,),
        )
        rows = cur.fetchall()
    if not rows:
        raise HTTPException(
            status_code=404,
            detail=f"No official Phase 7 portfolio for formation_date={formation_date} strategy={strategy}",
        )
    if len(rows) != len(TICKERS):
        raise HTTPException(
            status_code=500,
            detail=f"Official portfolio {run_id!r} has {len(rows)} holdings, expected {len(TICKERS)}",
        )
    w_series = pd.Series({ticker: float(weight) for ticker, weight in rows}).reindex(TICKERS)
    if w_series.isna().any():
        raise HTTPException(status_code=500, detail=f"Official portfolio {run_id!r} has a ticker mismatch")
    return w_series.to_numpy()


def _validate_weights(w: np.ndarray, strategy: str, per_weight_tol: float = 1e-6, sum_tol: float = 1e-4) -> None:
    """Trust-but-verify the already-persisted official weights (same
    invariants `optimizer.walkforward._validate_frozen_weights` enforced
    at construction time) — a violation here means the persisted Phase 7
    data is corrupted, an integrity issue (500), never a client error.

    `sum_tol` is looser than `per_weight_tol`: these are REAL persisted
    values read back from `portfolios.weight NUMERIC(8,6)` (schema.sql) —
    up to ~5e-7 storage-rounding per row, which can accumulate to ~2.5e-5
    across 50 rows. Same tolerance class already used for this exact
    check in tests/test_api_db_integration.py's
    `test_portfolio_real_combinations_satisfy_frozen_constraints`."""
    if not np.isfinite(w).all():
        raise HTTPException(status_code=500, detail=f"Official portfolio for {strategy!r} has non-finite weight(s)")
    if abs(float(np.sum(w)) - 1.0) > sum_tol:
        raise HTTPException(status_code=500, detail=f"Official portfolio for {strategy!r} weights sum to {np.sum(w):.6g}, expected 1")
    if w.min() < -per_weight_tol:
        raise HTTPException(status_code=500, detail=f"Official portfolio for {strategy!r} has a negative weight")
    if strategy != "EQUAL_WEIGHT" and w.max() > config.MAX_WEIGHT + per_weight_tol:
        raise HTTPException(status_code=500, detail=f"Official portfolio for {strategy!r} exceeds MAX_WEIGHT={config.MAX_WEIGHT}")


@router.get("/risk", response_model=RiskResponse)
def get_risk(
    formation_date: Optional[date_type] = Query(
        None,
        description="Phase 7 portfolio-eligible formation date (46 available, 2022-03-28 to 2026-01-02). Defaults to the latest.",
    ),
    strategy: str = Query("LW_MAXSHARPE", description="One of: " + ", ".join(VALID_STRATEGIES)),
    covariance: Optional[str] = Query(
        None,
        description=(
            "Covariance estimator, one of SAMPLE/LW. Ignored (must match the strategy's own "
            "construction estimator if given) for the four optimized strategies. For EQUAL_WEIGHT "
            "only, selects which historical covariance is used to ANALYZE the fixed 1/50 benchmark "
            "(defaults to LW) — never implies that estimator was used to construct it."
        ),
    ),
    conn=Depends(get_db),
) -> RiskResponse:
    if strategy not in VALID_STRATEGIES:
        raise HTTPException(status_code=400, detail=f"Unknown strategy {strategy!r}; expected one of {VALID_STRATEGIES}")
    if covariance is not None and covariance not in VALID_COVARIANCE_ESTIMATORS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown covariance estimator {covariance!r}; expected one of {VALID_COVARIANCE_ESTIMATORS}",
        )

    eligible_dates = _official_formation_dates(conn)
    if formation_date is None:
        if not eligible_dates:
            raise HTTPException(status_code=404, detail="No official Phase 7 portfolio data available")
        formation_date = eligible_dates[-1]
    elif formation_date not in eligible_dates:
        raise HTTPException(
            status_code=404,
            detail=f"formation_date={formation_date} is not a Phase 7 portfolio-eligible formation",
        )

    implied_key = _implied_covariance_key(strategy)
    if implied_key is not None:
        # Optimized strategy: covariance is determined by construction,
        # never a free selector — reject a disagreeing client value rather
        # than silently overriding it or silently ignoring it.
        if covariance is not None and covariance != implied_key:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"strategy={strategy!r} was constructed under {implied_key} covariance; "
                    f"covariance={covariance!r} would misrepresent its construction risk model"
                ),
            )
        covariance_key = implied_key
    else:
        # EQUAL_WEIGHT: a genuine risk-analysis choice, defaults to LW.
        covariance_key = covariance or "LW"

    w = _load_official_weights(conn, formation_date, strategy)
    _validate_weights(w, strategy)

    _, sigma_by_estimator, _ = reconstruct_mu_sigma_rf(conn, formation_date)
    sigma = sigma_by_estimator[covariance_key]

    sigma_p = portfolio_volatility(w, sigma)
    if not np.isfinite(sigma_p) or sigma_p <= 0:
        raise HTTPException(status_code=500, detail="Reconstructed predicted volatility is non-finite or non-positive")

    sigma_w = sigma @ w
    marginal_rc = sigma_w / sigma_p
    component_rc = w * marginal_rc
    if not (np.isfinite(marginal_rc).all() and np.isfinite(component_rc).all()):
        raise HTTPException(status_code=500, detail="Non-finite risk contribution computed")
    risk_share = component_rc / sigma_p

    rc_sum_error = abs(float(component_rc.sum()) - sigma_p)
    if rc_sum_error > 1e-4:
        raise HTTPException(status_code=500, detail=f"Component risk contributions do not sum to predicted volatility (error={rc_sum_error:.3e})")

    sectors_by_ticker = [_sector_for_ticker(t) for t in TICKERS]
    assets = [
        AssetRiskRow(
            ticker=TICKERS[i],
            sector=sectors_by_ticker[i],
            weight=float(w[i]),
            marginal_risk_contribution=float(marginal_rc[i]),
            component_risk_contribution=float(component_rc[i]),
            risk_share=float(risk_share[i]),
        )
        for i in range(len(TICKERS))
    ]

    sectors: List[SectorRiskRow] = []
    for sector_name in (_SECTOR_IT, _SECTOR_FIN):
        mask = np.array([s == sector_name for s in sectors_by_ticker])
        sector_weight = float(w[mask].sum())
        sector_rc = float(component_rc[mask].sum())
        sectors.append(
            SectorRiskRow(
                sector=sector_name,
                weight=sector_weight,
                component_risk_contribution=sector_rc,
                risk_share=sector_rc / sigma_p,
            )
        )

    strategy_info = STRATEGY_REGISTRY[strategy]
    largest_weight = float(w.max())
    hhi = float(np.sum(w * w))

    return RiskResponse(
        formation_date=formation_date,
        strategy=strategy_info,
        covariance_estimator="Sample" if covariance_key == "SAMPLE" else "Ledoit-Wolf",
        forecast_horizon_sessions=config.FORECAST_HORIZON,
        covariance_window_sessions=config.COVAR_WINDOW,
        portfolio=PortfolioRiskSummary(
            predicted_volatility_21=sigma_p,
            largest_weight=largest_weight,
            active_holdings_count=sum(1 for wi in w if wi > 1e-9),
            concentration_hhi=hhi,
            effective_holdings=1.0 / hhi,
            max_weight_constraint=config.MAX_WEIGHT if strategy_info.is_optimized else None,
        ),
        assets=assets,
        sectors=sectors,
        source="reconstructed_from_frozen_phase7_methodology",
    )
