"""
RiskFecta Phase 4A — leakage-safe walk-forward fold construction.

Single authoritative implementation of the formation/label-availability
boundary (ML_SPEC.md §8, §11-§14, §22; BUILD_PLAN.md Phase 4). This module
builds fold *metadata* and *row selections* only — it never fits a model,
so every fold's structure is independently testable without training
anything (task brief §4).

================================================================================
FORMATION / LABEL BOUNDARY — resolved explicitly (task brief §24)
================================================================================

Session indices are all 0-based positions in the GLOBAL trading-session
calendar (`build_global_calendar`): the sorted, de-duplicated union of every
valid-session date across the pooled 50-ticker universe. Because the real
Bloomberg universe is dense (62,800 = 50 tickers x 1,256 valid sessions each,
per README.md "Phase 1" status), the global calendar equals every ticker's
own per-ticker session sequence; a ticker with a genuine gap (a fixture, or a
future data revision) simply has no row at that global session for the
sessions it's missing — it is never fabricated (see "ticker isolation"
below).

- **Formation session T**: the session at which a pooled forecast is formed
  for every ticker. Informational cutoff = "information available as of the
  close of session T" (ML_SPEC.md §7's "formation date t", §22's "training
  window/formation date"). Close of T IS observable at T (T is itself a
  contemporaneously-valid feature session — ML_SPEC.md §6's "contemporaneously
  valid, i.e. known as of the formation date").
- **First OOS evaluation session**: T itself. The pooled model forecasts
  y(i, T) for every ticker using only features dated <= T; that forecast is
  scored 21 valid sessions later once TRI(T+21) is realized (ML_SPEC.md §14).
- **Candidate 252-session training-FEATURE window**: sessions
  `[T - TRAIN_WINDOW + 1, T]` inclusive (252 sessions ending at, and
  including, T — ML_SPEC.md §12's "trailing TRAIN_WINDOW valid sessions").
  A candidate row's FEATURES are always contemporaneously valid as of its
  own session s <= T <= T, so feature-date separation alone is satisfied by
  construction for every session in this window.
- **Label-eligibility boundary (the actual purge)**: a candidate training
  row at feature-session s is usable only if its OWN target's endpoint
  `s + FORECAST_HORIZON` is <= T — i.e. its label is already realized/
  observable as of the T cutoff. This is the ENDPOINT == T ALLOWED
  convention: a label ending exactly on T uses only information through the
  close of T, which is exactly what's available at the formation cutoff.
  `s + FORECAST_HORIZON > T` is PURGED (its label needs information from
  after T, which has not happened yet as of formation).
- **Final eligible training feature session**: `T - FORECAST_HORIZON`
  (because s + 21 <= T  <=>  s <= T - 21).
- **Usable labeled sessions per ticker before feature-availability
  filtering**: from `T - TRAIN_WINDOW + 1` through `T - FORECAST_HORIZON`
  inclusive =
      (T - FORECAST_HORIZON) - (T - TRAIN_WINDOW + 1) + 1
    = TRAIN_WINDOW - FORECAST_HORIZON
    = 252 - 21 = 231.
  (Feature-availability filtering — warm-up NaNs in RSI/MACD/Bollinger/
  momentum, missing macro joins — is applied on TOP of this and can only
  reduce the count further; see `training_rows_for_fold`.)

--------------------------------------------------------------------------------
Worked numeric example (task brief §24), T = 300:
--------------------------------------------------------------------------------
    Formation session:                       T = 300
    Training candidate FEATURE window:       [49, 300]   (252 sessions)
    Label-eligible training window:          [49, 279]   (231 sessions)
    Last eligible training feature session:  279
      -> target endpoint = 279 + 21 = 300;  is 300 <= 300?  YES -> ALLOWED
    First PURGED training feature session:   280
      -> target endpoint = 280 + 21 = 301;  is 301 <= 300?  NO  -> PURGED
    First OOS evaluation session:            300 (== T)
    Usable labeled sessions per ticker:      279 - 49 + 1 = 231
See tests/test_folds.py::test_formation_label_boundary_worked_example_T_300
for the executable form of this exact example.

If a future methodology revision instead wants the stricter "endpoint < T"
convention (excluding a label ending exactly on T), that is a ONE-LINE
change (`<=` -> `<` in `training_rows_for_fold`'s purge mask) with an
explicit, separately-approved methodology note — never silently alternated
between call sites (ML_SPEC.md §22 "each rolling-window step re-fits... a
single globally-fit scaler across all history is a leakage bug" — the same
"never silently alternate" discipline applies to this boundary).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import pandas as pd

import config
from pipeline.sessions import DATE_COL, TICKER_COL, valid_sessions
from pipeline.targets import TARGET_COL

GLOBAL_SESSION_COL = "global_session"


# ---------------------------------------------------------------------------
# Global (pooled) trading-session calendar
# ---------------------------------------------------------------------------
def build_global_calendar(prices: pd.DataFrame) -> pd.DatetimeIndex:
    """Sorted, de-duplicated union of every valid-session date across all
    tickers in `prices` (a prices_raw-shaped frame with ticker/date/close).
    This is the single shared session-index reference every fold's
    formation/training/eval sessions are expressed in (ML_SPEC.md §14's
    pooled "at each formation date t... for all 50 tickers").
    """
    sess = valid_sessions(prices)
    dates = pd.DatetimeIndex(sorted(pd.to_datetime(sess[DATE_COL]).unique()))
    return dates


def attach_global_session(df: pd.DataFrame, calendar: pd.DatetimeIndex) -> pd.DataFrame:
    """Attach a `global_session` column: each row's 0-based position in
    `calendar` for its own `date`. A row whose date is not in `calendar`
    (a genuine per-ticker gap) gets NaN — it is never assigned a
    neighboring/interpolated session index (ticker isolation: a ticker
    missing a session is simply absent from that global session, never
    fabricated from another ticker's calendar).
    """
    idx_map = pd.Series(range(len(calendar)), index=calendar)
    out = df.copy()
    out[GLOBAL_SESSION_COL] = pd.to_datetime(out[DATE_COL]).map(idx_map)
    return out


# ---------------------------------------------------------------------------
# Fold metadata
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Fold:
    fold_id: int
    formation_session: int          # T
    formation_date: pd.Timestamp
    train_window_first_session: int  # T - TRAIN_WINDOW + 1
    train_window_last_session: int   # T  (candidate FEATURE window end)
    label_eligible_last_session: int  # T - FORECAST_HORIZON (purge boundary)
    eval_session: int                # T (== formation_session)
    eval_date: pd.Timestamp
    target_realized_session: int     # T + FORECAST_HORIZON (when eval row's own target becomes known)

    @property
    def usable_labeled_sessions_before_feature_filtering(self) -> int:
        return self.label_eligible_last_session - self.train_window_first_session + 1


def fold_for_formation_session(
    T: int,
    calendar: pd.DatetimeIndex,
    train_window: int,
    horizon: int,
    fold_id: int = 0,
) -> Fold:
    """Pure constructor: the formation/label-boundary math for a single
    formation session T, factored out of `build_folds` so the worked
    numeric example (task brief §24) can be exercised directly for any T,
    independent of `step` alignment. Does not validate that T actually has
    a full candidate window or a realizable target within `calendar` —
    `build_folds` enforces those; this function is the pure arithmetic
    core only.
    """
    return Fold(
        fold_id=fold_id,
        formation_session=T,
        formation_date=calendar[T],
        train_window_first_session=T - train_window + 1,
        train_window_last_session=T,
        label_eligible_last_session=T - horizon,
        eval_session=T,
        eval_date=calendar[T],
        target_realized_session=T + horizon,
    )


def build_folds(
    calendar: pd.DatetimeIndex,
    train_window: int = None,
    step: int = None,
    horizon: int = None,
) -> List[Fold]:
    """Build the full walk-forward fold sequence over `calendar`.

    A formation session T is included only when:
      - a full `train_window`-session candidate FEATURE window exists
        (T >= train_window - 1), and
      - T's own OOS target is realizable within the given calendar
        (T + horizon <= len(calendar) - 1) — required for HISTORICAL
        walk-forward evaluation (ML_SPEC.md §14); a production/live
        forecast fold (target not yet realized) is a distinct, later use
        case not exercised in Phase 4A.

    Folds advance by exactly `step` valid sessions (never calendar days —
    `calendar` is itself session-indexed, so a plain integer step is
    automatically session-based, not calendar-based).
    """
    train_window = config.TRAIN_WINDOW if train_window is None else train_window
    step = config.STEP if step is None else step
    horizon = config.FORECAST_HORIZON if horizon is None else horizon
    if train_window <= horizon:
        raise ValueError("build_folds: train_window must exceed horizon (purge would empty every fold)")

    n = len(calendar)
    t_min = train_window - 1
    t_max = n - 1 - horizon
    folds: List[Fold] = []
    fold_id = 0
    for T in range(t_min, t_max + 1, step):
        folds.append(fold_for_formation_session(T, calendar, train_window, horizon, fold_id=fold_id))
        fold_id += 1
    return folds


# ---------------------------------------------------------------------------
# Row selection (purge-safe) — independently testable without any model.
# ---------------------------------------------------------------------------
def training_rows_for_fold(
    panel: pd.DataFrame,
    fold: Fold,
    feature_cols: Sequence[str],
    target_col: str = TARGET_COL,
) -> pd.DataFrame:
    """Purge-safe pooled training rows for `fold`.

    `panel` must already carry a `global_session` column (see
    `attach_global_session`) and one row per (ticker, date) with `feature_cols`
    and `target_col` populated where computable (NaN otherwise — never
    fabricated). Selection, in order:

      1. FEATURE-date window: train_window_first_session <= global_session
         <= train_window_last_session (== T). Every session in this range is
         contemporaneously valid by construction (features.py never uses
         future information), so this step alone would satisfy plain
         feature-date separation.
      2. LABEL-availability purge (the actual leakage guard, §3 of the task
         brief): global_session <= label_eligible_last_session
         (== T - FORECAST_HORIZON). This is strictly tighter than (1) and is
         what actually removes the final `horizon` sessions of the candidate
         window (their own target endpoint would exceed T).
      3. Target must be realized/non-NaN (defensive — purge boundary (2)
         already guarantees this for a dense calendar, but a genuine
         mid-window missing TRI still yields NaN and must not be silently
         imputed).
      4. All `feature_cols` non-NaN (missing-feature handling, task brief §8
         — no zero-fill, no bfill; a row lacking a required model input is
         simply excluded).

    Pooled across every ticker present in `panel` (task brief §6) — no
    per-ticker special-casing beyond what step 4 naturally does per row.
    """
    mask = (
        panel[GLOBAL_SESSION_COL].notna()
        & (panel[GLOBAL_SESSION_COL] >= fold.train_window_first_session)
        & (panel[GLOBAL_SESSION_COL] <= fold.label_eligible_last_session)
        & panel[target_col].notna()
    )
    for col in feature_cols:
        mask &= panel[col].notna()
    return panel.loc[mask].copy()


def eval_rows_for_fold(
    panel: pd.DataFrame,
    fold: Fold,
    feature_cols: Sequence[str],
) -> pd.DataFrame:
    """Purge-safe OOS evaluation rows for `fold`: exactly the ticker rows
    dated at `fold.eval_session` (== T) with all `feature_cols` present.
    The row's own `target_col` (if present in `panel`) is carried through
    unchanged for later scoring only — it is realized 21 sessions after T
    and must never be consulted before that point by any training or
    forecast-forming code path (only by post-hoc metrics computation).
    """
    mask = panel[GLOBAL_SESSION_COL] == fold.eval_session
    for col in feature_cols:
        mask &= panel[col].notna()
    return panel.loc[mask].copy()


# ---------------------------------------------------------------------------
# Internal temporal validation split (task brief §12-§13, §17-J)
# ---------------------------------------------------------------------------
def internal_validation_split(fold: Fold, val_sessions: int = None) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    """Split a fold's already-purged eligible training range
    `[train_window_first_session, label_eligible_last_session]` into
    (inner_train_range, inner_val_range) for hyperparameter/methodology
    selection (ML_SPEC.md §13):

      - inner_val is the chronologically LAST `val_sessions` sessions of the
        eligible range -> strictly after inner_train, never overlapping it.
      - inner_val's own last session is `label_eligible_last_session`
        (== T - horizon), which is strictly BEFORE the outer OOS session T
        -> outer-OOS-blind by construction (inner_val never reaches T).
      - Every session in both ranges already satisfies the fold's own
        label-availability purge (they're a subset of the eligible range),
        so internal validation inherits the same leakage guarantee, not a
        separately-invented one.

    Phase 4A does not exercise this to run a real hyperparameter search
    (see models/xgboost_model.py, models/baselines.py docstrings) — it is
    implemented and tested now so the scaffolding exists and is provably
    correct before any real tuning is authorized.
    """
    val_sessions = config.STEP if val_sessions is None else val_sessions
    lo, hi = fold.train_window_first_session, fold.label_eligible_last_session
    inner_val_start = hi - val_sessions + 1
    if inner_val_start <= lo:
        raise ValueError(
            "internal_validation_split: val_sessions too large for this fold's eligible training range "
            f"(lo={lo}, hi={hi}, val_sessions={val_sessions})"
        )
    inner_train_range = (lo, inner_val_start - 1)
    inner_val_range = (inner_val_start, hi)
    return inner_train_range, inner_val_range


def rows_in_session_range(
    panel: pd.DataFrame,
    session_range: Tuple[int, int],
    feature_cols: Sequence[str],
    target_col: str = TARGET_COL,
) -> pd.DataFrame:
    """Pooled rows whose `global_session` falls within `session_range`
    (inclusive), with target and all `feature_cols` present. Shared helper
    for slicing an inner_train_range / inner_val_range from
    `internal_validation_split` — no new purge logic invented here, it
    reuses the same non-NaN-target / non-NaN-feature eligibility rule as
    `training_rows_for_fold`.
    """
    lo, hi = session_range
    mask = (
        panel[GLOBAL_SESSION_COL].notna()
        & (panel[GLOBAL_SESSION_COL] >= lo)
        & (panel[GLOBAL_SESSION_COL] <= hi)
        & panel[target_col].notna()
    )
    for col in feature_cols:
        mask &= panel[col].notna()
    return panel.loc[mask].copy()
