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

from app.routes import frontier, health, market, models, portfolios, predictions, prices, risk, universe

app = FastAPI(
    title="RiskFecta API",
    description=(
        "Read-only API over Supabase-hosted PostgreSQL: historical market data "
        "(Phase 1), frozen Phase 4-6 walk-forward forecast cross-sections, the "
        "frozen, official Phase 7 historical portfolio-construction experiment, "
        "a reconstructed (never persisted) Phase 7 efficient frontier, and "
        "formation-time portfolio risk-contribution analytics."
    ),
    version="0.5.0",
)

# ---------------------------------------------------------------------------
# CORS (TRD.md §9 — prepared for the Phase 2B/2C React frontend).
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


# ---------------------------------------------------------------------------
# Error handling (TRD.md §16) — a database failure (connection refused, DNS,
# auth) must produce a consistent 503 JSON body and must NEVER surface the
# underlying psycopg2 message, which can include host/user/connection info.
# ---------------------------------------------------------------------------
@app.exception_handler(psycopg2.Error)
async def psycopg2_error_handler(request: Request, exc: psycopg2.Error) -> JSONResponse:
    return JSONResponse(status_code=503, content={"detail": "Database unavailable"})


app.include_router(health.router)
app.include_router(universe.router)
app.include_router(prices.router)
app.include_router(market.router)
app.include_router(predictions.router)
app.include_router(models.router)
app.include_router(portfolios.router)
app.include_router(frontier.router)
app.include_router(risk.router)
