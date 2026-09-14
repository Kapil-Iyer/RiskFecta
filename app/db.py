"""
Thin FastAPI-facing wrapper around the existing Phase 1B connection helper.

Deliberately does NOT fork a second connection stack: every request opens its
connection via `pipeline.db.get_connection()` (the same psycopg2 helper used
by ingestion/tests), scoped to the request lifetime by the `get_db`
dependency below. No SQLAlchemy, no pooling layer — a new connection per
request is the intentionally minimal Phase 2A approach (see final report).
"""
from __future__ import annotations

from typing import Iterator

import psycopg2.extensions

from pipeline import db as pipeline_db


def get_db() -> Iterator[psycopg2.extensions.connection]:
    """FastAPI dependency: yields a request-scoped psycopg2 connection.

    If `pipeline_db.get_connection()` itself raises (e.g. the database is
    unreachable), that exception propagates to the app-level psycopg2 error
    handler in `app.main` — never leaked as a route-specific stack trace.
    """
    conn = pipeline_db.get_connection()
    try:
        yield conn
    finally:
        conn.close()


def check_connectivity() -> bool:
    """Best-effort DB reachability check for the readiness endpoint.

    Never raises and never surfaces the underlying exception (which could
    include host/port details) — callers only ever see True/False.
    """
    try:
        conn = pipeline_db.get_connection()
    except Exception:
        return False
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT 1")
            cur.fetchone()
        return True
    except Exception:
        return False
    finally:
        conn.close()
