"""
RiskFecta FastAPI application — Phase 2A skeleton.

Thin HTTP layer per TRD.md §9: request validation (Pydantic), calling into
`pipeline/` or querying PostgreSQL directly via `pipeline.db`, and response
serialization. No ML/optimizer business logic lives here (TRD.md §2, §8) —
that stays in `pipeline/`, `models/`, `optimizer/`.

Run locally:
    uvicorn app.main:app --reload

Environment variables:
    DATABASE_URL   Required (see config.get_database_url) — Supabase/Postgres
                    connection string, loaded from .env, never hardcoded.
    CORS_ORIGINS    Optional, comma-separated list of allowed frontend
                    origins. Defaults to the standard local React dev-server
                    origins (CRA: 3000, Vite: 5173). Set explicitly in any
                    non-local environment — never wildcarded.
"""
from __future__ import annotations

import os

import psycopg2
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from app.routes import backtest, frontier, health, market, models, portfolios, predictions, prices, risk, universe

app = FastAPI(
    title="RiskFecta API",
    description=(
        "Read-only API over Supabase-hosted PostgreSQL: historical market data "
        "(Phase 1), frozen Phase 4-6 walk-forward forecast cross-sections, the "
        "frozen, official Phase 7 historical portfolio-construction experiment, "
        "a reconstructed (never persisted) Phase 7 efficient frontier, "
        "formation-time portfolio risk-contribution analytics, and the frozen "
        "Phase 7 walk-forward historical evidence across 46 evaluation periods."
    ),
    version="0.6.0",
)

# ---------------------------------------------------------------------------
# Error handling (TRD.md §16) — a database failure (connection refused, DNS,
# auth) must produce a consistent 503 JSON body and must NEVER surface the
# underlying psycopg2 message, which can include host/user/connection info.
# ---------------------------------------------------------------------------
@app.exception_handler(psycopg2.Error)
async def psycopg2_error_handler(request: Request, exc: psycopg2.Error) -> JSONResponse:
    return JSONResponse(status_code=503, content={"detail": "Database unavailable"})


class _UnhandledErrorMiddleware(BaseHTTPMiddleware):
    """Turns any exception with no registered `@app.exception_handler` (e.g.
    a route hitting a genuinely missing local file) into a normal 500
    JSONResponse — deliberately as MIDDLEWARE, not `@app.exception_handler
    (Exception)`.

    Starlette special-cases a handler registered for the bare `Exception`
    class (or status 500): it runs inside `ServerErrorMiddleware`, which
    Starlette places OUTSIDE every `add_middleware`-registered middleware,
    including CORSMiddleware below. A response built there never passes
    back through CORSMiddleware, so it leaves with no
    `Access-Control-Allow-Origin` header. A browser treats a cross-origin
    response with no CORS header as a network failure, not an HTTP error —
    this is why the deployed frontend showed "Unable to reach the
    RiskFecta API" for what was actually a real backend 500 (Phase 8F
    production-remediation finding).

    A plain middleware has no such special case: `add_middleware` PREPENDS
    to Starlette's middleware list, so registering this one BEFORE
    CORSMiddleware (below) places CORSMiddleware OUTSIDE it. Catching the
    exception here and returning a normal Response means CORSMiddleware
    sees an ordinary successful call to the next layer and adds its header
    exactly as it would for any other response.

    The body is deliberately generic (same reasoning as the psycopg2
    handler above): the underlying exception could include a local
    filesystem path or other internal detail that must never reach the
    client.
    """

    async def dispatch(self, request: Request, call_next):
        try:
            return await call_next(request)
        except Exception:
            return JSONResponse(
                status_code=500,
                content={"detail": "Research data could not be loaded due to a temporary server issue. Please retry in a moment."},
            )


app.add_middleware(_UnhandledErrorMiddleware)

# ---------------------------------------------------------------------------
# CORS (TRD.md §9 — prepared for the Phase 2B/2C React frontend). Registered
# AFTER _UnhandledErrorMiddleware above so it ends up OUTSIDE it (see that
# class's docstring) and can attach headers to the 500s it produces.
# ---------------------------------------------------------------------------
_DEFAULT_DEV_ORIGINS = "http://localhost:3000,http://localhost:5173"
_allowed_origins = [
    origin.strip()
    for origin in os.environ.get("CORS_ORIGINS", _DEFAULT_DEV_ORIGINS).split(",")
    if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_credentials=False,
    allow_methods=["GET"],
    allow_headers=["*"],
)


app.include_router(health.router)
app.include_router(universe.router)
app.include_router(prices.router)
app.include_router(market.router)
app.include_router(predictions.router)
app.include_router(models.router)
app.include_router(portfolios.router)
app.include_router(frontier.router)
app.include_router(risk.router)
app.include_router(backtest.router)
