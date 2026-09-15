"""
RiskFecta Phase 5B — real historical LSTM walk-forward orchestration
(ML_SPEC.md §12-§14, §17-§18, §22; BUILD_PLAN.md Phase 5; "Continue
RiskFecta 2.0 Phase 5B" task brief).

Wires together, per outer Phase 4 fold (`pipeline.folds.build_folds` —
UNCHANGED, no LSTM-specific outer calendar is invented):

  1. `pipeline.sequences.build_sequence_index` (built ONCE over full
     history, shared across all folds)
  2. `pipeline.lstm_folds.build_lstm_fold_sequences` (Reading A fold
     wiring, locked)
  3. `pipeline.scaling.fit_sequence_scaler` / `transform_sequences`
     — fit on `inner_train` ONLY (task brief §7, stricter than the
     Phase 5A audit's looser "training data" language — the Phase 5B
     lock is authoritative)
  4. `models.lstm.train_lstm` — early stopping on `inner_val` only
  5. `models.lstm.predict_lstm` — on `outer_oos` only, for scoring

Nothing here fits a scaler or trains a model on `outer_oos`. Nothing here
computes an ensemble. Nothing here touches March 2026 data — the panel is
built from `prices_raw`/`features`, whose real content ends ~2026-02-27
and which this module never filters by date (BUILD_PLAN.md Phase 9 gate).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import pandas as pd

import config
from pipeline.folds import Fold, build_folds, build_global_calendar
from pipeline.lstm_folds import LSTMFoldSequences, build_lstm_fold_sequences
from pipeline.scaling import fit_sequence_scaler, transform_sequences
from pipeline.sequences import (
    LSTM_FEATURE_COLS,
    SEQ_END_DATE_COL,
    SEQ_TARGET_COL,
    SEQ_TICKER_COL,
    build_sequence_index,
    extract_sequence_array,
)
from pipeline.sessions import DATE_COL, TICKER_COL
from pipeline.targets import TARGET_COL, compute_targets
from models.lstm import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_DROPOUT,
    DEFAULT_EARLY_STOPPING_PATIENCE,
    DEFAULT_HIDDEN_SIZE,
    DEFAULT_LEARNING_RATE,
    DEFAULT_MAX_EPOCHS,
    DEFAULT_NUM_LAYERS,
    DEFAULT_SEED,
    predict_lstm,
    train_lstm,
)

LOAD_PANEL_SQL = """
SELECT p.ticker, p.date, p.open, p.high, p.low, p.close, p.volume, p.total_return_idx,
       f.rsi_14, f.macd, f.macd_signal, f.bb_upper, f.bb_lower, f.volatility_20d,
       f.momentum_3m, f.momentum_6m
FROM prices_raw p
JOIN features f ON f.ticker = p.ticker AND f.date = p.date
ORDER BY p.ticker, p.date
"""


def load_lstm_panel(conn) -> pd.DataFrame:
    """Read-only load of the real prices_raw + features join, plus the
    real target_21d (`pipeline.targets.compute_targets`, computed here
    exactly as Phase 3/4 do — no independent target formula). Never
    filters by date -- March exclusion is a property of what the terminal
    Bloomberg export actually contains (~2026-02-27), not a WHERE clause
    this function adds. Never writes anything."""
    panel = pd.read_sql(LOAD_PANEL_SQL, conn)
    panel[DATE_COL] = pd.to_datetime(panel[DATE_COL])
    numeric_cols = ["open", "high", "low", "close", "volume", "total_return_idx"] + [
        c for c in LSTM_FEATURE_COLS if c not in ("open", "high", "low", "close", "volume")
    ]
    for col in numeric_cols:
        panel[col] = pd.to_numeric(panel[col], errors="coerce")

    targets = compute_targets(panel[[TICKER_COL, DATE_COL, "close", "total_return_idx"]])
    panel = panel.merge(targets, on=[TICKER_COL, DATE_COL], how="left")
    return panel


@dataclass
class FoldRunResult:
    fold: Fold
    n_train: int
    n_inner_train: int
    n_inner_val: int
    n_outer_oos: int
    epochs_trained: int
    stopped_early: bool
    best_val_loss: float
    predictions: pd.DataFrame  # ticker, forecast_date, target_date, lstm_pred, global_session


@dataclass
class WalkforwardResult:
    calendar: pd.DatetimeIndex
    folds: List[Fold]
    fold_results: List[FoldRunResult] = field(default_factory=list)

    @property
    def all_predictions(self) -> pd.DataFrame:
        if not self.fold_results:
            return pd.DataFrame(columns=[TICKER_COL, "forecast_date", "target_date", "lstm_pred"])
        return pd.concat([fr.predictions for fr in self.fold_results], ignore_index=True)


def run_fold(
    panel: pd.DataFrame,
    seq_index: pd.DataFrame,
    calendar: pd.DatetimeIndex,
    fold: Fold,
    hidden_size: int = DEFAULT_HIDDEN_SIZE,
    num_layers: int = DEFAULT_NUM_LAYERS,
    dropout: float = DEFAULT_DROPOUT,
    lr: float = DEFAULT_LEARNING_RATE,
    batch_size: int = DEFAULT_BATCH_SIZE,
    max_epochs: int = DEFAULT_MAX_EPOCHS,
    patience: int = DEFAULT_EARLY_STOPPING_PATIENCE,
    seed: int = DEFAULT_SEED,
    val_sessions: Optional[int] = None,
) -> FoldRunResult:
    """Run ONE fresh pooled LSTM for ONE outer fold: fold-wiring -> scaler
    (inner_train-only) -> train (internal-validation early stopping) ->
    predict (outer_oos only). No warm-start from any other fold — a brand
    new `PooledLSTM` is constructed inside `train_lstm` every call."""
    fs: LSTMFoldSequences = build_lstm_fold_sequences(seq_index, fold, val_sessions=val_sessions)

    X_inner_train = extract_sequence_array(panel, fs.inner_train)
    X_inner_val = extract_sequence_array(panel, fs.inner_val)
    X_outer_oos = extract_sequence_array(panel, fs.outer_oos)

    scaler = fit_sequence_scaler(X_inner_train)  # task brief §7: inner_train ONLY, never full .train
    X_inner_train_s = transform_sequences(X_inner_train, scaler)
    X_inner_val_s = transform_sequences(X_inner_val, scaler)
    X_outer_oos_s = transform_sequences(X_outer_oos, scaler)

    y_inner_train = fs.inner_train[SEQ_TARGET_COL].to_numpy(dtype=float)
    y_inner_val = fs.inner_val[SEQ_TARGET_COL].to_numpy(dtype=float)

    result = train_lstm(
        X_inner_train_s, y_inner_train, X_inner_val_s, y_inner_val,
        hidden_size=hidden_size, num_layers=num_layers, dropout=dropout,
        lr=lr, batch_size=batch_size, max_epochs=max_epochs, patience=patience, seed=seed,
    )
    preds = predict_lstm(result.model, X_outer_oos_s)

    target_date = calendar[fold.target_realized_session]
    pred_df = fs.outer_oos[[SEQ_TICKER_COL, SEQ_END_DATE_COL]].copy()
    pred_df = pred_df.rename(columns={SEQ_TICKER_COL: TICKER_COL, SEQ_END_DATE_COL: "forecast_date"})
    pred_df["target_date"] = target_date
    pred_df["lstm_pred"] = preds
    pred_df["global_session"] = fs.outer_oos["global_session"].to_numpy()

    best_val_loss = (
        result.val_losses[result.best_epoch] if result.best_epoch >= 0 and result.val_losses else float("nan")
    )
    return FoldRunResult(
        fold=fold,
        n_train=len(fs.train),
        n_inner_train=len(fs.inner_train),
        n_inner_val=len(fs.inner_val),
        n_outer_oos=len(fs.outer_oos),
        epochs_trained=len(result.train_losses),
        stopped_early=result.stopped_early,
        best_val_loss=best_val_loss,
        predictions=pred_df,
    )


def run_walkforward(
    panel: pd.DataFrame,
    val_sessions: Optional[int] = None,
    seed: int = DEFAULT_SEED,
    progress_cb=None,
) -> WalkforwardResult:
    """Run the frozen LSTM walk-forward across every outer Phase 4 fold
    (`pipeline.folds.build_folds`, UNCHANGED config: TRAIN_WINDOW=252,
    STEP=21, FORECAST_HORIZON=21). `progress_cb(i, n, fold)`, if given, is
    called before each fold runs (for a human-readable progress log only —
    no effect on any computation)."""
    calendar = build_global_calendar(panel[[TICKER_COL, DATE_COL, "close"]])
    folds = build_folds(calendar)
    seq_index = build_sequence_index(panel, calendar=calendar)

    wf = WalkforwardResult(calendar=calendar, folds=folds)
    for i, fold in enumerate(folds):
        if progress_cb is not None:
            progress_cb(i, len(folds), fold)
        fr = run_fold(panel, seq_index, calendar, fold, seed=seed, val_sessions=val_sessions)
        wf.fold_results.append(fr)
    return wf
