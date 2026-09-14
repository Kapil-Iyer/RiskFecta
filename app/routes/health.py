"""Liveness + readiness endpoints (TRD.md §16 — observability/error handling)."""
from __future__ import annotations

from fastapi import APIRouter, Response

from app.db import check_connectivity
from app.schemas import HealthResponse, ReadinessResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Process liveness only — deliberately never touches the database, so
    this always answers even if PostgreSQL is unreachable."""
    return HealthResponse(status="ok")


@router.get("/health/ready", response_model=ReadinessResponse)
def readiness(response: Response) -> ReadinessResponse:
    """Readiness: verifies PostgreSQL connectivity without leaking credentials
    or connection details. Returns HTTP 503 (with a body still describing the
    failure state) when the database is unreachable."""
    if check_connectivity():
        return ReadinessResponse(status="ok", database="connected")
    response.status_code = 503
    return ReadinessResponse(status="error", database="unavailable")
