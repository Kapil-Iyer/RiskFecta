"""
RiskFecta Phase 5A — pooled LSTM sequence construction (ML_SPEC.md §9, §17,
§18; BUILD_PLAN.md Phase 5).

Sequence construction only. No model, no scaler, no fold/training-set
wiring lives here (see models/lstm.py, pipeline/scaling.py) — kept
independently testable without training anything, mirroring the separation
already established by pipeline/folds.py (row selection) vs.
models/xgboost_model.py (fitting) in Phase 4A.

================================================================================
LOCKED LSTM FEATURE SURFACE (Phase 5A task brief §8 — resolved, not invented)
================================================================================
`config.LSTM_ALL_FEATURES` = `config.LSTM_PRICE_COLS + config.LSTM_FEATURE_COLS`
= OHLCV + {rsi_14, macd, macd_signal, bb_upper, bb_lower, volatility_20d,
momentum_3m, momentum_6m} (13 columns). This is the frozen list already
defined in config.py (the single source of truth) and matches ML_SPEC.md
§6/§17's "price/technical" language for the LSTM branch — it is NOT the
5-column XGBoost macro/technical subset (config.XGBOOST_FEATURE_COLS, which
adds vix/yield_10y and is a deliberately different, smaller tabular
surface). No static snapshot field (beta, mkt_cap_log, div_yield), no
sector, and no ticker identity is ever part of this surface
(`_assert_locked_lstm_feature_surface` below is defense in depth).

================================================================================
MISSING-DATA ELIGIBILITY POLICY (task brief §10 — resolved)
================================================================================
A candidate 60-session window is eligible only if EVERY one of its 60 rows
has EVERY required feature column non-NaN. No zero-fill, no forward-fill,
no backward-fill, no global fill is ever applied here — a row that lacks a
required feature (Phase 3 technical-indicator warm-up NaNs are the expected
source, e.g. the first ~20-125 sessions of each ticker's history) simply
disqualifies every 60-session window that would need it, exactly the same
"missing input -> excluded, never fabricated" discipline already used by
`pipeline.folds.training_rows_for_fold` (its step 4: "All feature_cols
non-NaN ... no zero-fill, no bfill; a row lacking a required model input is
simply excluded").

================================================================================
DECISION GATE — RESOLVED (Reading A, locked by the user; see
pipeline/lstm_folds.py)
================================================================================
`TRAIN_WINDOW=252` bounds only a sequence sample's own END session
(analogous to how `pipeline.folds` bounds an XGBoost training row's own
date, while that row's technical features — e.g. `momentum_6m`'s own
126-session backward lookback — are allowed to reach earlier than the
252-session window's start, because the window bounds the ROW's date, not
the ROW's own internal computation depth). A sample's 60-session causal
INPUT may reach earlier than the 252-session candidate window when that
history genuinely exists — this module never fabricates it (no
negative-index wraparound, no padding, no fill; `build_sequence_index`
only ever emits a window backed by 60 real, same-ticker, feature-complete
observations, so a sample requiring nonexistent pre-history is simply
absent from its output).

This module implements the two things that are pure sequence-construction
concerns, independent of any OUTER fold:
  (a) causal, per-ticker sequence construction (`build_sequence_index`,
      `extract_sequence_array`); and
  (b) sample-level LABEL availability (`label_eligible_mask`) — the
      target-realization boundary, which only depends on a sample's own END
      session vs. the outer formation cutoff T.

The fold-wiring layer that selects, for a given outer `pipeline.folds.Fold`,
which `build_sequence_index` rows are pooled-training / internal-validation
/ outer-OOS under Reading A lives in `pipeline.lstm_folds` (reusing
`pipeline.folds.training_rows_for_fold` / `eval_rows_for_fold` /
`internal_validation_split` verbatim — no purge logic duplicated there
either).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

import config
from pipeline.folds import GLOBAL_SESSION_COL
from pipeline.sessions import DATE_COL, TICKER_COL, valid_sessions
from pipeline.targets import TARGET_COL

SEQUENCE_LENGTH = config.LSTM_SEQ
LSTM_FEATURE_COLS = list(config.LSTM_ALL_FEATURES)

SEQ_TICKER_COL = TICKER_COL
SEQ_START_DATE_COL = "seq_start_date"
SEQ_END_DATE_COL = "seq_end_date"
SEQ_START_POS_COL = "seq_start_pos"   # ticker-local 0-based position (own valid-session sequence)
SEQ_END_POS_COL = "seq_end_pos"       # ticker-local 0-based position == sample's formation session
SEQ_TARGET_COL = "target_21d"

SEQUENCE_INDEX_COLUMNS = [
    SEQ_TICKER_COL, SEQ_START_DATE_COL, SEQ_END_DATE_COL,
    SEQ_START_POS_COL, SEQ_END_POS_COL, SEQ_TARGET_COL, GLOBAL_SESSION_COL,
]

# Forbidden static-snapshot / identity columns (ML_SPEC.md §10, §22) —
# defense in depth, mirrors models/xgboost_model.py's
# _assert_locked_feature_surface for the LSTM branch.
_FORBIDDEN_LSTM_FEATURES = {
    "ticker_id", "beta", "mkt_cap_log", "sector", "div_yield",
    "cur_mkt_cap", "beta_raw_overridable", "dividend_indicated_yield",
    "vix", "yield_10y",
}


def _assert_locked_lstm_feature_surface(feature_cols: Sequence[str]) -> None:
    leaked = _FORBIDDEN_LSTM_FEATURES.intersection(feature_cols)
    if leaked:
        raise ValueError(f"pipeline.sequences: forbidden feature(s) present: {sorted(leaked)}")


# ---------------------------------------------------------------------------
# Sequence index construction
# ---------------------------------------------------------------------------
def _row_complete_mask(df: pd.DataFrame, feature_cols: Sequence[str]) -> pd.Series:
    """True where every required feature is present for that single row.
    Never imputes — purely descriptive of the row as given."""
    return df[list(feature_cols)].notna().all(axis=1)


def build_sequence_index(
    panel: pd.DataFrame,
    feature_cols: Optional[Sequence[str]] = None,
    seq_len: Optional[int] = None,
    target_col: str = TARGET_COL,
    calendar: Optional[pd.DatetimeIndex] = None,
) -> pd.DataFrame:
    """Build the metadata index of every eligible 60-valid-session (default
    `config.LSTM_SEQ`) causal sequence in `panel` (a features-table-shaped
    frame: one row per (ticker, date) with `feature_cols` and, optionally,
    `target_col` populated where computable).

    Eligibility (per ticker, independently — no cross-ticker interaction of
    any kind, task brief §4/§7):
      - `panel` is first reduced to valid trading sessions only
        (`pipeline.sessions.valid_sessions`) and sorted ascending by date
        per ticker — this is what makes "60 sessions" a SESSION count, never
        a calendar-day span (task brief §4, §20-D).
      - A window ending at ticker-local position `i` (0-based, that
        ticker's own `i`-th valid session) is eligible only if positions
        `[i - seq_len + 1, i]` all have every required feature present (see
        module docstring, "Missing-data eligibility policy"). This mirrors
        the same backward-looking `rolling(window, min_periods=window)`
        idiom already used by `pipeline.features.compute_bollinger` /
        `compute_volatility` — a window "ending at t" only ever reads
        `t - window + 1 .. t`, never `t+1` or later (task brief §20-E).

    Each eligible sample's target (`target_21d`, task brief §20-F) is taken
    directly from `panel[target_col]` at the window's OWN end row — exactly
    `TRI(end+21)/TRI(end)-1` per ML_SPEC.md §8, already computed upstream by
    `pipeline.targets.compute_targets` and merged into `panel` by the
    caller. A target may legitimately be NaN (the sample's own +21 horizon
    has not yet realized, or its own TRI is missing) — sequence
    ELIGIBILITY never depends on target availability (task brief §5: input
    causality and label availability are separate concerns); only training
    USE of a sample depends on the target (see `label_eligible_mask`).

    If `calendar` (a pooled global session calendar,
    `pipeline.folds.build_global_calendar`) is supplied, each sample's own
    END date is also resolved to its `global_session` position in that
    pooled calendar — needed downstream to compare a sample's own formation
    session against an outer fold's formation cutoff T
    (`label_eligible_mask`). Without `calendar`, `global_session` is left
    NaN (pure per-ticker sequence construction needs no pooled calendar at
    all — task brief §9 keeps sequence construction independently
    testable).
    """
    feature_cols = list(feature_cols) if feature_cols is not None else LSTM_FEATURE_COLS
    _assert_locked_lstm_feature_surface(feature_cols)
    seq_len = SEQUENCE_LENGTH if seq_len is None else seq_len
    if seq_len <= 0:
        raise ValueError("build_sequence_index: seq_len must be a positive valid-session count")

    sess = valid_sessions(panel)
    has_target = target_col in sess.columns

    global_session_map: Optional[pd.Series] = None
    if calendar is not None:
        global_session_map = pd.Series(range(len(calendar)), index=calendar)

    rows: List[dict] = []
    for ticker, g in sess.groupby(TICKER_COL, sort=False):
        g = g.sort_values(DATE_COL, kind="mergesort").reset_index(drop=True)
        complete = _row_complete_mask(g, feature_cols).astype(int)
        # Backward-looking window sum: position i's value is
        # sum(complete[i-seq_len+1 .. i]) once i >= seq_len-1 (min_periods
        # enforces that, giving NaN before then -> naturally ineligible).
        window_complete_count = complete.rolling(window=seq_len, min_periods=seq_len).sum()
        eligible_end_positions = np.flatnonzero((window_complete_count == seq_len).to_numpy())

        dates = pd.to_datetime(g[DATE_COL]).to_numpy()
        targets = g[target_col].to_numpy() if has_target else None

        for end_pos in eligible_end_positions:
            start_pos = end_pos - seq_len + 1
            end_date = pd.Timestamp(dates[end_pos])
            row = {
                SEQ_TICKER_COL: ticker,
                SEQ_START_DATE_COL: pd.Timestamp(dates[start_pos]),
                SEQ_END_DATE_COL: end_date,
                SEQ_START_POS_COL: int(start_pos),
                SEQ_END_POS_COL: int(end_pos),
                SEQ_TARGET_COL: (targets[end_pos] if has_target else np.nan),
                GLOBAL_SESSION_COL: (
                    global_session_map.get(end_date, np.nan) if global_session_map is not None else np.nan
                ),
            }
            rows.append(row)

    if not rows:
        return pd.DataFrame(columns=SEQUENCE_INDEX_COLUMNS)

    out = pd.DataFrame(rows, columns=SEQUENCE_INDEX_COLUMNS)
    return out.sort_values([SEQ_TICKER_COL, SEQ_END_DATE_COL], kind="mergesort").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Array materialization — pure extraction, no eligibility logic.
# ---------------------------------------------------------------------------
def extract_sequence_array(
    panel: pd.DataFrame,
    seq_index: pd.DataFrame,
    feature_cols: Optional[Sequence[str]] = None,
) -> np.ndarray:
    """Materialize the `(n_samples, seq_len, n_features)` float array for
    every row of `seq_index` (as produced by `build_sequence_index` over
    the SAME `panel` — this function trusts the index and does not
    re-derive or re-validate eligibility itself, mirroring
    `models.xgboost_model`'s trust of `pipeline.folds`'s already-purged
    rows).

    Extraction is by ticker-local position (`seq_start_pos`/`seq_end_pos`),
    never by date arithmetic, so it is immune to calendar gaps by
    construction (task brief §20-D).
    """
    feature_cols = list(feature_cols) if feature_cols is not None else LSTM_FEATURE_COLS
    _assert_locked_lstm_feature_surface(feature_cols)

    if seq_index.empty:
        return np.empty((0, 0, len(feature_cols)), dtype=float)

    sess = valid_sessions(panel)
    ticker_matrices: Dict[str, np.ndarray] = {}
    for ticker, g in sess.groupby(TICKER_COL, sort=False):
        g = g.sort_values(DATE_COL, kind="mergesort")
        ticker_matrices[ticker] = g[feature_cols].to_numpy(dtype=float)

    seq_len = int(seq_index[SEQ_END_POS_COL].iloc[0] - seq_index[SEQ_START_POS_COL].iloc[0] + 1)
    n = len(seq_index)
    out = np.empty((n, seq_len, len(feature_cols)), dtype=float)
    for row_i, rec in enumerate(seq_index.itertuples(index=False)):
        ticker = getattr(rec, SEQ_TICKER_COL)
        start_pos = getattr(rec, SEQ_START_POS_COL)
        end_pos = getattr(rec, SEQ_END_POS_COL)
        mat = ticker_matrices[ticker]
        out[row_i] = mat[start_pos : end_pos + 1, :]
    return out


# ---------------------------------------------------------------------------
# Label availability (task brief §5-B, §20-G) — independent of the
# TRAIN_WINDOW/LSTM_SEQ decision gate (see module docstring).
# ---------------------------------------------------------------------------
def label_eligible_mask(
    seq_index: pd.DataFrame,
    formation_session: int,
    horizon: Optional[int] = None,
) -> pd.Series:
    """A sequence sample is TRAINING-eligible at outer formation cutoff
    `formation_session` (T, a pooled `global_session` position — see
    `pipeline.folds.Fold.formation_session`) only if its own target
    endpoint `global_session + horizon` is `<= T` — the exact same
    ENDPOINT-==-T-ALLOWED convention as
    `pipeline.folds.training_rows_for_fold` (boundary exactly on T is
    allowed; one session after T is purged, task brief §20-G).

    Requires `seq_index[GLOBAL_SESSION_COL]` to be populated (i.e.
    `build_sequence_index` was called with a `calendar`) — label
    availability is inherently a POOLED, cross-ticker-comparable concept (T
    is a pooled formation session), so a purely per-ticker sequence index
    cannot answer this question.

    This function resolves ONLY the label-availability constraint. It does
    NOT decide, and its result must not be read as deciding, whether a
    given sample also falls inside a fold's 252-session TRAIN_WINDOW
    feature-reach — that is the still-open decision gate (module
    docstring).
    """
    horizon = config.FORECAST_HORIZON if horizon is None else horizon
    if seq_index[GLOBAL_SESSION_COL].isna().any():
        raise ValueError(
            "label_eligible_mask: seq_index has NaN global_session — "
            "build_sequence_index must be called with a `calendar` first"
        )
    return (seq_index[GLOBAL_SESSION_COL] + horizon) <= formation_session
