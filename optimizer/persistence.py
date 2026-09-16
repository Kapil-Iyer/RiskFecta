"""
RiskFecta Phase 7A — `portfolios`/`risk_metrics` persistence
(optimizer/persistence.py). Mock-tested only.

Mirrors `pipeline/predictions.py`'s persistence pattern (transactional,
`execute_values`-based, table-scoped) with one structural difference the
frozen schema forces: `portfolios` and `risk_metrics` (schema.sql Tables
4/5) have NO UNIQUE constraint on (run_id, ticker) or
(run_id, metric_name, ticker) — only a bare `id SERIAL PRIMARY KEY`.
There is therefore no natural ON CONFLICT target; persistence here is a
plain INSERT (append-only). "Idempotence" for this module means only that
re-running with the SAME `run_id` and DataFrame inserts the SAME rows
again — a real, documented limitation of the frozen schema, not silently
papered over here. A schema amendment adding a UNIQUE(run_id, ticker)
constraint would be the correct fix for true upsert idempotence, but that
is out of scope for Phase 7A (task brief §17: "Do NOT change schema unless
the frozen architecture genuinely requires it").

NOT called against the live database anywhere in Phase 7A — implemented
and mock-tested only. Real `portfolios`/`risk_metrics` must remain at 0
rows until the Cursor Integrity Audit passes and real execution is
explicitly approved.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from psycopg2.extras import execute_values

from pipeline import db
from pipeline.sessions import TICKER_COL

RUN_ID_COL = "run_id"
WEIGHT_COL = "weight"
TARGET_RETURN_COL = "target_return"
PORTFOLIO_VOL_COL = "portfolio_vol"
SHARPE_RATIO_COL = "sharpe_ratio"

PORTFOLIOS_TABLE_COLUMNS = [
    RUN_ID_COL, TICKER_COL, WEIGHT_COL, TARGET_RETURN_COL, PORTFOLIO_VOL_COL, SHARPE_RATIO_COL,
]

PORTFOLIOS_INSERT_SQL = """
INSERT INTO portfolios (run_id, ticker, weight, target_return, portfolio_vol, sharpe_ratio)
VALUES %s
"""

METRIC_NAME_COL = "metric_name"
METRIC_VALUE_COL = "metric_value"

RISK_METRICS_TABLE_COLUMNS = [RUN_ID_COL, METRIC_NAME_COL, METRIC_VALUE_COL, TICKER_COL]

RISK_METRICS_INSERT_SQL = """
INSERT INTO risk_metrics (run_id, metric_name, metric_value, ticker)
VALUES %s
"""


def build_portfolio_rows(
    run_id: str, tickers, weights, target_return, portfolio_vol, sharpe_ratio
) -> pd.DataFrame:
    """One row per ticker for a single optimized portfolio (schema.sql
    Table 4: one row per (run_id, ticker) weight, with the portfolio-level
    target_return/portfolio_vol/sharpe_ratio repeated on every row —
    exactly the shape the frozen 5-column table supports; no new columns
    invented)."""
    tickers = list(tickers)
    weights = np.asarray(weights, dtype=float)
    if len(tickers) != len(weights):
        raise ValueError("build_portfolio_rows: tickers/weights length mismatch")
    return pd.DataFrame({
        RUN_ID_COL: run_id,
        TICKER_COL: tickers,
        WEIGHT_COL: weights,
        TARGET_RETURN_COL: target_return,
        PORTFOLIO_VOL_COL: portfolio_vol,
        SHARPE_RATIO_COL: sharpe_ratio,
    })[PORTFOLIOS_TABLE_COLUMNS]


def build_risk_metric_rows(run_id: str, metrics: dict, ticker: Optional[str] = None) -> pd.DataFrame:
    """`metrics`: {metric_name: metric_value}. `ticker=None` -> a
    portfolio-level row (schema.sql: `ticker` NULL for portfolio-level,
    per-asset rows pass a real ticker instead)."""
    rows = [
        {RUN_ID_COL: run_id, METRIC_NAME_COL: name, METRIC_VALUE_COL: value, TICKER_COL: ticker}
        for name, value in metrics.items()
    ]
    return pd.DataFrame(rows, columns=RISK_METRICS_TABLE_COLUMNS)


def _to_sql_value(v):
    """Same NaN/NaT/None -> SQL NULL normalization as pipeline.features/predictions._to_sql_value."""
    if v is None:
        return None
    if isinstance(v, float) and np.isnan(v):
        return None
    if v is pd.NaT:
        return None
    try:
        if pd.isna(v):
            return None
    except (TypeError, ValueError):
        pass
    return v


def _records(df: pd.DataFrame, columns) -> List[Tuple]:
    out = df[columns].copy()
    records = []
    for row in out.itertuples(index=False, name=None):
        records.append(tuple(_to_sql_value(v) for v in row))
    return records


def insert_portfolios(conn, df: pd.DataFrame) -> int:
    """Low-level append-only INSERT into `portfolios`. Does NOT commit —
    caller controls the transaction boundary (see `persist_portfolios`)."""
    records = _records(df, PORTFOLIOS_TABLE_COLUMNS)
    if not records:
        return 0
    with conn.cursor() as cur:
        execute_values(cur, PORTFOLIOS_INSERT_SQL, records, page_size=1000)
    return len(records)


def insert_risk_metrics(conn, df: pd.DataFrame) -> int:
    """Low-level append-only INSERT into `risk_metrics`. Does NOT commit —
    caller controls the transaction boundary (see `persist_risk_metrics`)."""
    records = _records(df, RISK_METRICS_TABLE_COLUMNS)
    if not records:
        return 0
    with conn.cursor() as cur:
        execute_values(cur, RISK_METRICS_INSERT_SQL, records, page_size=1000)
    return len(records)


@dataclass
class PersistResult:
    rows_prepared: int
    rows_in_table_after: int


def persist_portfolios(conn=None, df: Optional[pd.DataFrame] = None) -> PersistResult:
    """NOT called against the live database anywhere in Phase 7A —
    implemented and mock-tested only, pending the Cursor Integrity Audit
    and explicit real-execution approval. Real `portfolios` must remain
    entirely empty until then."""
    if df is None:
        raise ValueError("persist_portfolios: df is required (no default real portfolio computation here)")
    owns_conn = conn is None
    conn = conn or db.get_connection()
    try:
        with conn:
            n = insert_portfolios(conn, df)
        with conn:
            rows_after = db.fetch_scalar(conn, "SELECT COUNT(*) FROM portfolios")
        return PersistResult(rows_prepared=n, rows_in_table_after=int(rows_after))
    finally:
        if owns_conn:
            conn.close()


def persist_risk_metrics(conn=None, df: Optional[pd.DataFrame] = None) -> PersistResult:
    """Same contract as `persist_portfolios`, for `risk_metrics`. NOT
    called against the live database anywhere in Phase 7A."""
    if df is None:
        raise ValueError("persist_risk_metrics: df is required (no default real risk-metric computation here)")
    owns_conn = conn is None
    conn = conn or db.get_connection()
    try:
        with conn:
            n = insert_risk_metrics(conn, df)
        with conn:
            rows_after = db.fetch_scalar(conn, "SELECT COUNT(*) FROM risk_metrics")
        return PersistResult(rows_prepared=n, rows_in_table_after=int(rows_after))
    finally:
        if owns_conn:
            conn.close()


# ---------------------------------------------------------------------------
# Phase 7B — experiment/run identity (task brief §17-18).
#
# `run_id VARCHAR(50)` is the ONLY identity column schema.sql gives
# `portfolios`/`risk_metrics` beyond `ticker` — there is no typed
# formation-date or strategy column, and no UNIQUE constraint at all
# (Phase 7A finding, restated: persistence here is append-only INSERT).
# Rather than changing schema (task brief §17-H: "Do not silently change
# schema"), this encodes (experiment_id, strategy, formation_date) into
# one deterministic, parseable `run_id` string — schema.sql's own header
# comment already documents `run_id` as "a logical/application key only
# (no FK)", which is exactly this usage pattern. `ticker` (already a real
# column) remains the per-row identity within one run_id.
#
# Collision detection (`run_id_exists`/`assert_run_id_available`) is a
# read-before-write check at the APPLICATION level — schema.sql gives no
# UNIQUE constraint to enforce it at the database level, so it is not
# atomic against a genuine race between two concurrent writers. That is
# an accepted, documented limitation for a single-operator research
# pipeline (not a production multi-writer system) — never silently
# described as a database-level guarantee.
# ---------------------------------------------------------------------------
RUN_ID_MAX_LEN = 50
VALID_STRATEGIES = ("SAMPLE_MINVOL", "SAMPLE_MAXSHARPE", "LW_MINVOL", "LW_MAXSHARPE", "EQUAL_WEIGHT")


def build_run_id(experiment_id: str, strategy: str, formation_date) -> str:
    """Deterministic `f"{experiment_id}_{strategy}_{YYYY-MM-DD}"`. Fails
    loudly rather than silently truncating/reinterpreting on: an unknown
    strategy, an `experiment_id` containing `_` (would break
    `parse_run_id`'s positional split) or whitespace, or a result
    exceeding schema.sql's `VARCHAR(50)` limit."""
    if strategy not in VALID_STRATEGIES:
        raise ValueError(f"build_run_id: unknown strategy {strategy!r}, expected one of {VALID_STRATEGIES}")
    if not experiment_id or any(ch.isspace() for ch in experiment_id):
        raise ValueError("build_run_id: experiment_id must be non-empty and contain no whitespace")
    if "_" in experiment_id:
        raise ValueError("build_run_id: experiment_id must not contain '_' (breaks run_id parsing)")
    date_str = pd.Timestamp(formation_date).strftime("%Y-%m-%d")
    run_id = f"{experiment_id}_{strategy}_{date_str}"
    if len(run_id) > RUN_ID_MAX_LEN:
        raise ValueError(f"build_run_id: run_id {run_id!r} ({len(run_id)} chars) exceeds VARCHAR({RUN_ID_MAX_LEN})")
    return run_id


def parse_run_id(run_id: str) -> dict:
    """Inverse of `build_run_id`. Fails loudly if `run_id` doesn't decode
    to a known strategy (e.g. was never produced by `build_run_id`)."""
    parts = run_id.split("_")
    if len(parts) < 3:
        raise ValueError(f"parse_run_id: {run_id!r} does not have the expected experiment_strategy_date shape")
    experiment_id = parts[0]
    date_str = parts[-1]
    strategy = "_".join(parts[1:-1])
    if strategy not in VALID_STRATEGIES:
        raise ValueError(f"parse_run_id: could not recover a known strategy from {run_id!r} (got {strategy!r})")
    return {"experiment_id": experiment_id, "strategy": strategy, "formation_date": date_str}


def run_id_exists(conn, table: str, run_id: str) -> bool:
    """Read-only collision check. `table` must be `portfolios` or
    `risk_metrics` — never any other table."""
    if table not in ("portfolios", "risk_metrics"):
        raise ValueError(f"run_id_exists: unsupported table {table!r}")
    with conn.cursor() as cur:
        cur.execute(f"SELECT 1 FROM {table} WHERE run_id = %s LIMIT 1", (run_id,))
        return cur.fetchone() is not None


def assert_run_id_available(conn, table: str, run_id: str) -> None:
    """§17-E/F: check BEFORE any insert; fail loudly rather than silently
    append a duplicate official run."""
    if run_id_exists(conn, table, run_id):
        raise ValueError(
            f"assert_run_id_available: run_id {run_id!r} already exists in {table} — refusing to "
            "silently append a duplicate official run (existing rows are never destructively overwritten)"
        )


def persist_portfolio_run(conn, run_id: str, df: pd.DataFrame) -> PersistResult:
    """Collision-safe persistence for ONE strategy's ONE formation-date
    portfolio: aborts before any row is inserted if `run_id` already
    exists in `portfolios` (§17-G: never destructively overwrite). NOT
    called against the live database anywhere in Phase 7B."""
    assert_run_id_available(conn, "portfolios", run_id)
    return persist_portfolios(conn=conn, df=df)


def persist_risk_metrics_run(conn, run_id: str, df: pd.DataFrame) -> PersistResult:
    """Same collision-safe contract as `persist_portfolio_run`, for
    `risk_metrics`."""
    assert_run_id_available(conn, "risk_metrics", run_id)
    return persist_risk_metrics(conn=conn, df=df)
