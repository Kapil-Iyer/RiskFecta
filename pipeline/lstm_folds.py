"""
RiskFecta Phase 5A — LSTM fold wiring (ML_SPEC.md §12-§14, §17-§18;
BUILD_PLAN.md Phase 5; "Continue RiskFecta 2.0 Phase 5A" task brief §1-§6).

================================================================================
DECISION GATE — RESOLVED (Reading A, locked by the user)
================================================================================
`TRAIN_WINDOW=252` bounds each LSTM training SAMPLE'S OWN END SESSION only
— exactly like it bounds an XGBoost training row's own date in
`pipeline.folds`. For outer formation session T, candidate sample END
sessions are `[T-251, T]`; the existing label-availability rule
(`s + FORECAST_HORIZON <= T`, endpoint exactly on T allowed) then narrows
this to label-eligible END sessions `[T-251, T-21]`. A sample's 60-session
causal INPUT may reach earlier than `T-251` — that earlier history is
context only, never an additional training target, and this module never
re-derives or re-checks it: `pipeline.sequences.build_sequence_index`
already guarantees every row in its output has exactly 60 REAL,
same-ticker, non-fabricated observations (no negative-index wraparound, no
padding, no fill) — a sample whose 60-session reach would need
nonexistent history is simply ABSENT from `seq_index` to begin with, so
this module needs no separate "does the history exist" check of its own.

================================================================================
ZERO DUPLICATED PURGE LOGIC
================================================================================
Every selection below is a direct call into the SAME `pipeline.folds`
functions Phase 4A already uses for XGBoost, applied to a sequence index
(`pipeline.sequences.build_sequence_index` output, which carries
`GLOBAL_SESSION_COL` = each sample's own END session) instead of a raw
feature-row panel:

  - `pipeline.folds.training_rows_for_fold`     -> candidate-window
    ([T-251,T]) + label-availability ([T-251,T-21]) + target-notna
    filtering — reused VERBATIM, not reimplemented.
  - `pipeline.folds.eval_rows_for_fold`         -> outer-OOS selection,
    exactly the fold's own formation/eval session T.
  - `pipeline.folds.internal_validation_split` /
    `pipeline.folds.rows_in_session_range`      -> internal temporal
    validation, unchanged.

`feature_cols=[]` is passed to every reused call because a sequence-index
row carries no raw feature COLUMNS of its own (those live in the
separately-materialized `(n, 60, k)` array via
`pipeline.sequences.extract_sequence_array`) — only metadata
(ticker/dates/positions/target/global_session). Passing `[]` makes each
reused function's feature-notna step a no-op, which is correct: a
sequence-index row already exists only because all 60 of its underlying
rows are feature-complete (`pipeline.sequences.build_sequence_index`'s own
eligibility rule) — re-checking feature completeness here would be
duplicated, not new, logic.

No LSTM is fit here. No array is materialized here, no scaler is fit
here. This module answers ONLY "which rows of `seq_index` belong in which
bucket for this fold" — array extraction is
`pipeline.sequences.extract_sequence_array`, scaling is
`pipeline.scaling`, training is `models.lstm.train_lstm`. Kept separate
exactly like `pipeline.folds` / `models.xgboost_model` are kept separate
in Phase 4A.

================================================================================
OUTER-OOS BLINDNESS IS STRUCTURAL, NOT CONVENTIONAL
================================================================================
`LSTMFoldSequences.outer_oos` can never overlap `.train` / `.inner_train` /
`.inner_val`: outer-OOS requires `global_session == fold.eval_session`
(== T), while every training/internal-validation row requires
`global_session <= fold.label_eligible_last_session` (== T - horizon) —
strictly less than T whenever `horizon > 0` (enforced by
`pipeline.folds.build_folds`'s own `train_window > horizon` check). A
caller that only ever passes `.inner_train`/`.inner_val` into
`models.lstm.train_lstm` (which itself has no outer-OOS parameter at all)
cannot leak `.outer_oos` into training or early stopping even by mistake.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd

from pipeline.folds import (
    Fold,
    eval_rows_for_fold,
    internal_validation_split,
    rows_in_session_range,
    training_rows_for_fold,
)
from pipeline.sequences import SEQ_TARGET_COL


@dataclass(frozen=True)
class LSTMFoldSequences:
    """Every `seq_index` subset needed to run one outer fold's pooled LSTM
    step. Rows only — no arrays, no scaler, no model."""

    fold: Fold
    train: pd.DataFrame        # candidate-window + label-eligible pooled training samples (pre-internal-split)
    inner_train: pd.DataFrame  # internal-temporal-validation TRAIN subset
    inner_val: pd.DataFrame    # internal-temporal-validation VAL subset (early-stopping signal ONLY)
    outer_oos: pd.DataFrame    # exactly the fold's own formation/eval session T, per ticker — scoring only


def build_lstm_fold_sequences(
    seq_index: pd.DataFrame,
    fold: Fold,
    val_sessions: Optional[int] = None,
) -> LSTMFoldSequences:
    """Select this fold's pooled LSTM training / internal-validation /
    outer-OOS sequence-index subsets.

    `seq_index` must already carry a populated `global_session` column
    (`pipeline.sequences.build_sequence_index` called with a `calendar`) —
    the pooled formation-session concept this function reasons about (T)
    only exists relative to a shared, cross-ticker calendar.
    """
    train = training_rows_for_fold(seq_index, fold, feature_cols=[], target_col=SEQ_TARGET_COL)
    inner_train_range, inner_val_range = internal_validation_split(fold, val_sessions=val_sessions)
    inner_train = rows_in_session_range(seq_index, inner_train_range, feature_cols=[], target_col=SEQ_TARGET_COL)
    inner_val = rows_in_session_range(seq_index, inner_val_range, feature_cols=[], target_col=SEQ_TARGET_COL)
    outer_oos = eval_rows_for_fold(seq_index, fold, feature_cols=[])
    return LSTMFoldSequences(fold=fold, train=train, inner_train=inner_train, inner_val=inner_val, outer_oos=outer_oos)
