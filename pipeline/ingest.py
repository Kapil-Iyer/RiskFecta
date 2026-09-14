"""
RiskFecta Phase 1B — ingestion of Phase 1A-normalized Bloomberg data into
Supabase-hosted PostgreSQL.

Scope (locked, see BUILD_PLAN.md Phase 1, and the Phase 1B task brief):
- Reuses pipeline.validate / pipeline.normalize verbatim — the session-
  validity gate (PX_LAST only, never TOTAL_RETURN_INDEX — ML_SPEC.md §3) and
  the NULL-preserving normalization logic are NOT re-derived here.
- Only `prices_raw` is ingested. The frozen 5-table schema.sql provides no
  authorized destination for the raw macro series or static snapshot
  fields as their own tables (see the Phase 1B report for the full
  discrepancy writeup) — those remain available as normalized in-memory
  DataFrames via pipeline.normalize only, not persisted to Postgres in this
  subphase.
- Idempotent by construction: INSERT ... ON CONFLICT (ticker, date) DO
  UPDATE against prices_raw's own UNIQUE(ticker, date) constraint, so
  re-running ingestion against the same source data converges to the same
  table state rather than duplicating rows.
- No fillna/zero-fill/forward-fill/back-fill/interpolation anywhere in this
  module. Genuine missing values arrive as NaN from pipeline.normalize and
  are passed through as SQL NULL.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import pandas as pd
from psycopg2.extras import execute_values

from pipeline import db, normalize

PRICE_COLUMNS = ["ticker", "date", "open", "high", "low", "close", "volume", "total_return_idx"]

PRICE_UPSERT_SQL = """
INSERT INTO prices_raw (ticker, date, open, high, low, close, volume, total_return_idx)
VALUES %s
ON CONFLICT (ticker, date) DO UPDATE SET
    open = EXCLUDED.open,
    high = EXCLUDED.high,
    low = EXCLUDED.low,
    close = EXCLUDED.close,
    volume = EXCLUDED.volume,
    total_return_idx = EXCLUDED.total_return_idx
"""


def cast_volume_to_bigint(volume: pd.Series) -> pd.Series:
    """Validate every non-null PX_VOLUME value is mathematically integral and
    cast to pandas' nullable ``Int64`` dtype (BIGINT-compatible), preserving
    genuine missing values as <NA> (-> SQL NULL).

    Never rounds a genuinely fractional value — raises ValueError instead, so
    a malformed source value fails loudly rather than being silently coerced.
    """
    s = pd.to_numeric(volume, errors="raise")
    non_null = s.dropna()
    non_integral = non_null[non_null != non_null.round()]
    if not non_integral.empty:
        raise ValueError(
            f"cast_volume_to_bigint: {len(non_integral)} genuinely fractional PX_VOLUME "
            f"value(s) found (e.g. {non_integral.iloc[0]!r}) — refusing to silently round"
        )
    return s.round().astype("Int64")


def _prepare_prices_for_insert(df: pd.DataFrame) -> List[Tuple]:
    """Normalized prices DataFrame -> list of DB-ready row tuples.

    - volume: validated + cast to nullable Int64 (see cast_volume_to_bigint).
    - date: pandas Timestamp -> datetime.date (psycopg2-adaptable).
    - Every NaN/<NA> becomes Python None (-> SQL NULL); no other value is
      touched or fabricated.
    """
    out = df[PRICE_COLUMNS].copy()
    out["volume"] = cast_volume_to_bigint(out["volume"])
    out["date"] = pd.to_datetime(out["date"]).dt.date

    is_na = out.isna()
    obj = out.astype(object)
    obj = obj.where(~is_na, None)
    return list(obj.itertuples(index=False, name=None))


def upsert_prices(conn, df: pd.DataFrame) -> int:
    """Low-level UPSERT of an already-normalized prices DataFrame into
    ``prices_raw``. Does NOT commit — the caller controls the transaction
    boundary (see ``ingest_prices`` for the commit-on-success wrapper).
    """
    records = _prepare_prices_for_insert(df)
    if not records:
        return 0
    with conn.cursor() as cur:
        execute_values(cur, PRICE_UPSERT_SQL, records, page_size=1000)
    return len(records)


@dataclass
class IngestResult:
    rows_prepared: int
    rows_in_table_after: int


def ingest_prices(conn=None, normalized: Optional[pd.DataFrame] = None) -> IngestResult:
    """Ingest the Phase 1A-normalized, session-filtered price panel into
    ``prices_raw``.

    Idempotent: keyed on ``prices_raw``'s own ``UNIQUE(ticker, date)``
    constraint via ``ON CONFLICT ... DO UPDATE`` — re-running against
    unchanged source data converges to the same table state instead of
    duplicating logical observations.

    Runs inside a single transaction: any failure (e.g. a constraint
    violation) rolls back the entire batch — no partial writes.
    """
    owns_conn = conn is None
    conn = conn or db.get_connection()
    if normalized is None:
        normalized = normalize.normalize_prices()

    try:
        with conn:  # psycopg2: commits this block on clean exit, rolls back on exception
            n = upsert_prices(conn, normalized)
        with conn:
            rows_after = db.fetch_scalar(conn, "SELECT COUNT(*) FROM prices_raw")
        return IngestResult(rows_prepared=n, rows_in_table_after=int(rows_after))
    finally:
        if owns_conn:
            conn.close()
