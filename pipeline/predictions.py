"""
RiskFecta Phase 4A — `predictions`-table persistence (schema.sql Table 3).

Mirrors `pipeline/features.py`'s persistence pattern exactly (idempotent
UPSERT keyed on the table's own UNIQUE constraint, no new tables/columns,
transactional). Uses ONLY existing `predictions` columns:

    ticker, forecast_date, target_date, lstm_pred, xgb_pred, ensemble_pred,
    actual_return, directional_correct   (UNIQUE (ticker, forecast_date))

Baseline forecasts (historical-mean, momentum, Ridge) have NO dedicated
column in the frozen schema (only `xgb_pred`/`lstm_pred`/`ensemble_pred`
exist) — task brief §15: "persist only if existing columns clearly support
them; otherwise keep them as in-memory/evaluation artifacts... do not force
every baseline into the DB." Accordingly this module only ever writes
`xgb_pred` (plus ticker/forecast_date/target_date identity); baseline
forecasts are evaluation-only and are never passed to `build_predictions_rows`
or persisted here. `lstm_pred`/`ensemble_pred` are left NULL — Phase 4A
never computes them (LSTM is Phase 5, ensemble is Phase 6).

IMPORTANT (Phase 4A pre-audit gate, task brief §15, §18, §25): this module
is implemented and mock-tested (tests/test_predictions_persistence.py) but
`persist_predictions` is NOT called against the real, live database in
Phase 4A. The live `predictions` table must remain at 0 rows until the
Cursor Integrity Audit passes and real execution is explicitly approved.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from psycopg2.extras import execute_values

from pipeline import db
from pipeline.sessions import DATE_COL, TICKER_COL

FORECAST_DATE_COL = "forecast_date"
TARGET_DATE_COL = "target_date"
XGB_PRED_COL = "xgb_pred"

PREDICTIONS_TABLE_COLUMNS = [
    TICKER_COL, FORECAST_DATE_COL, TARGET_DATE_COL,
    "lstm_pred", XGB_PRED_COL, "ensemble_pred", "actual_return", "directional_correct",
]

PREDICTIONS_UPSERT_SQL = """
INSERT INTO predictions (
    ticker, forecast_date, target_date, lstm_pred, xgb_pred, ensemble_pred,
    actual_return, directional_correct
) VALUES %s
ON CONFLICT (ticker, forecast_date) DO UPDATE SET
    target_date = EXCLUDED.target_date,
    xgb_pred = EXCLUDED.xgb_pred
"""
# NOTE: only xgb_pred (+ identity) is updated on conflict — Phase 4A never
# writes lstm_pred/ensemble_pred/actual_return/directional_correct, so a
# re-run must not clobber values a later phase (5/6) may have already
# written for the same (ticker, forecast_date) row. This is a deliberate,
# narrower ON CONFLICT clause than features.py's (which owns every column
# of its table); Phase 5/6 add their own narrow UPDATE SET clauses for
# their own columns rather than widening this one.

# ---------------------------------------------------------------------------
# Phase 5A — LSTM narrow upsert (task brief §17, §20-R). Mirrors
# PREDICTIONS_UPSERT_SQL's narrow-ON-CONFLICT discipline exactly, but for
# `lstm_pred` instead of `xgb_pred`: on conflict, ONLY `target_date` and
# `lstm_pred` are updated, so an LSTM upsert can never null out
# xgb_pred/ensemble_pred/actual_return/directional_correct that Phase
# 4/6 already wrote for the same (ticker, forecast_date) row.
# ---------------------------------------------------------------------------
LSTM_PRED_COL = "lstm_pred"

LSTM_PREDICTIONS_UPSERT_SQL = """
INSERT INTO predictions (
    ticker, forecast_date, target_date, lstm_pred, xgb_pred, ensemble_pred,
    actual_return, directional_correct
) VALUES %s
ON CONFLICT (ticker, forecast_date) DO UPDATE SET
    target_date = EXCLUDED.target_date,
    lstm_pred = EXCLUDED.lstm_pred
"""


def build_predictions_rows(
    eval_rows: pd.DataFrame,
    xgb_preds: pd.Series,
    horizon: int,
) -> pd.DataFrame:
    """`eval_rows` (pipeline.folds.eval_rows_for_fold output, one row per
    ticker at the fold's formation date) + `xgb_preds` (models.xgboost_model
    .predict_xgboost output, indexed by ticker) -> a predictions-table-shaped
    DataFrame: forecast_date = the eval row's own date, target_date = that
    date's own trading-session position + `horizon` valid sessions later
    is NOT recomputed here (that would need the global calendar) — callers
    that need an exact target_date should pass one via a calendar lookup;
    this function accepts a pre-resolved `target_date` per row through
    `eval_rows` when present, else leaves target_date NaT rather than
    guessing a calendar-day offset (never `date + 21 calendar days`, which
    would silently reintroduce a calendar/session error — ML_SPEC.md §3,
    §29).
    """
    out = eval_rows[[TICKER_COL, DATE_COL]].copy()
    out = out.rename(columns={DATE_COL: FORECAST_DATE_COL})
    if TARGET_DATE_COL in eval_rows.columns:
        out[TARGET_DATE_COL] = eval_rows[TARGET_DATE_COL].values
    else:
        out[TARGET_DATE_COL] = pd.NaT
    out[XGB_PRED_COL] = out[TICKER_COL].map(xgb_preds)
    out["lstm_pred"] = None
    out["ensemble_pred"] = None
    out["actual_return"] = None
    out["directional_correct"] = None
    return out[PREDICTIONS_TABLE_COLUMNS]


def build_lstm_predictions_rows(
    eval_rows: pd.DataFrame,
    lstm_preds: pd.Series,
    horizon: int,
) -> pd.DataFrame:
    """Same shape/semantics as `build_predictions_rows`, but populates
    `lstm_pred` instead of `xgb_pred`. `xgb_pred`/`ensemble_pred`/
    `actual_return`/`directional_correct` are left `None` in the INSERT
    payload — irrelevant on conflict (LSTM_PREDICTIONS_UPSERT_SQL never
    touches them on an existing row); correctly NULL only for a genuinely
    NEW row, where Phase 4/6 simply haven't written those columns yet.

    Phase 5A NOTE (task brief §17, §21): this function is implemented and
    mock-tested only. It is never called with real Bloomberg-derived
    predictions, and `persist_lstm_predictions` is never called against the
    live database, in Phase 5A.
    """
    out = eval_rows[[TICKER_COL, DATE_COL]].copy()
    out = out.rename(columns={DATE_COL: FORECAST_DATE_COL})
    if TARGET_DATE_COL in eval_rows.columns:
        out[TARGET_DATE_COL] = eval_rows[TARGET_DATE_COL].values
    else:
        out[TARGET_DATE_COL] = pd.NaT
    out[LSTM_PRED_COL] = out[TICKER_COL].map(lstm_preds)
    out[XGB_PRED_COL] = None
    out["ensemble_pred"] = None
    out["actual_return"] = None
    out["directional_correct"] = None
    return out[PREDICTIONS_TABLE_COLUMNS]


def upsert_lstm_predictions(conn, df: pd.DataFrame) -> int:
    """Low-level narrow UPSERT into `predictions` touching only
    `lstm_pred` (+ identity/`target_date`) on conflict. Does NOT commit —
    caller controls the transaction boundary (see `persist_lstm_predictions`)."""
    records = _prepare_predictions_for_insert(df)
    if not records:
        return 0
    with conn.cursor() as cur:
        execute_values(cur, LSTM_PREDICTIONS_UPSERT_SQL, records, page_size=1000)
    return len(records)


def persist_lstm_predictions(conn=None, df: Optional[pd.DataFrame] = None) -> PersistResult:
    """Same transactional/idempotent contract as `persist_predictions`,
    using the narrow LSTM upsert (`LSTM_PREDICTIONS_UPSERT_SQL`) so
    existing `xgb_pred`/`actual_return`/`ensemble_pred`/
    `directional_correct` values are always preserved.

    NOT called against the live database anywhere in Phase 5A (task brief
    §17, §21) — implemented and mock-tested only, pending the Cursor
    Integrity Audit AND the still-open TRAIN_WINDOW/LSTM_SEQ decision gate
    (see `pipeline.sequences` module docstring).
    """
    if df is None:
        raise ValueError("persist_lstm_predictions: df is required (no default real prediction computation here)")
    owns_conn = conn is None
    conn = conn or db.get_connection()
    try:
        with conn:
            n = upsert_lstm_predictions(conn, df)
        with conn:
            rows_after = db.fetch_scalar(conn, "SELECT COUNT(*) FROM predictions")
        return PersistResult(rows_prepared=n, rows_in_table_after=int(rows_after))
    finally:
        if owns_conn:
            conn.close()


def _to_sql_value(v):
    """Same NaN/NaT/None -> SQL NULL normalization as pipeline.features._to_sql_value."""
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


def _prepare_predictions_for_insert(df: pd.DataFrame) -> List[Tuple]:
    out = df[PREDICTIONS_TABLE_COLUMNS].copy()
    for date_col in (FORECAST_DATE_COL, TARGET_DATE_COL):
        out[date_col] = pd.to_datetime(out[date_col]).dt.date
    records = []
    for row in out.itertuples(index=False, name=None):
        records.append(tuple(_to_sql_value(v) for v in row))
    return records


def upsert_predictions(conn, df: pd.DataFrame) -> int:
    """Low-level UPSERT into `predictions`. Does NOT commit — caller
    controls the transaction boundary (see persist_predictions)."""
    records = _prepare_predictions_for_insert(df)
    if not records:
        return 0
    with conn.cursor() as cur:
        execute_values(cur, PREDICTIONS_UPSERT_SQL, records, page_size=1000)
    return len(records)


@dataclass
class PersistResult:
    rows_prepared: int
    rows_in_table_after: int


def persist_predictions(conn=None, df: Optional[pd.DataFrame] = None) -> PersistResult:
    """Ingest an already-built predictions DataFrame into `predictions`.
    Idempotent (UNIQUE(ticker, forecast_date) + narrow ON CONFLICT DO
    UPDATE on xgb_pred/target_date only); transactional; touches only
    `predictions` — never prices_raw, features, portfolios, or risk_metrics.

    NOT called against the live database anywhere in Phase 4A (see module
    docstring) — implemented and mock-tested only, pending the Cursor
    Integrity Audit.
    """
    if df is None:
        raise ValueError("persist_predictions: df is required (no default real prediction computation here)")
    owns_conn = conn is None
    conn = conn or db.get_connection()
    try:
        with conn:
            n = upsert_predictions(conn, df)
        with conn:
            rows_after = db.fetch_scalar(conn, "SELECT COUNT(*) FROM predictions")
        return PersistResult(rows_prepared=n, rows_in_table_after=int(rows_after))
    finally:
        if owns_conn:
            conn.close()
