"""
RiskFecta Phase 4A — leakage-safe MVP baselines (ML_SPEC.md §15).

All three baselines consume the ALREADY purge-safe rows returned by
`pipeline.folds.training_rows_for_fold` / `eval_rows_for_fold` — none of
them re-derive or loosen the label-availability boundary themselves. They
add no new leakage surface of their own; each one is deliberately simple
mechanics (task brief §15: "no arbitrarily complicated baseline
construction").

Momentum-baseline lock (Phase 4A delta, post conditional-pass audit):
official baseline = 3-month momentum (`momentum_3m`); 6-month momentum
(`momentum_6m`) is a secondary diagnostic only, never a second official
baseline. See `momentum_baseline()` below for the single source of truth.
"""
from __future__ import annotations

from typing import Sequence, Tuple

import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from pipeline.sessions import TICKER_COL
from pipeline.targets import TARGET_COL

# ---------------------------------------------------------------------------
# A. Historical-mean / naive baseline
# ---------------------------------------------------------------------------
def historical_mean_baseline(training_rows: pd.DataFrame, target_col: str = TARGET_COL) -> pd.Series:
    """Per-ticker forecast = the mean of that ticker's OWN label-eligible
    21-session forward returns within the fold's TRAIN_WINDOW (ML_SPEC.md
    §15: "computed only from the same TRAIN_WINDOW of history available at
    t, no future information") — NOT a global mean over the full dataset,
    and NOT computed from `training_rows` unless the caller already scoped
    it to one fold's purge-safe window (`pipeline.folds.training_rows_for_fold`).
    This function performs no purging itself; it trusts its input is already
    purge-safe and only aggregates, so it can add no new leakage of its own.

    Returns a Series indexed by ticker; a ticker absent from `training_rows`
    (e.g. it has zero eligible labeled sessions this fold) is simply absent
    from the result — never fabricated as 0.0.
    """
    return training_rows.groupby(TICKER_COL)[target_col].mean()


# ---------------------------------------------------------------------------
# B. Momentum baseline
# ---------------------------------------------------------------------------
# Decision LOCKED (Phase 4A momentum-baseline delta, post Cursor conditional-
# pass audit): ML_SPEC.md §15's "prior 3-/6-month return" is resolved as
# follows —
#   - OFFICIAL_MOMENTUM_BASELINE_COL ("momentum_3m") is THE momentum
#     baseline: what `momentum_baseline()` defaults to, and what any Phase
#     4A comparison table/report treats as "the" momentum baseline model.
#   - SECONDARY_MOMENTUM_DIAGNOSTIC_COL ("momentum_6m") remains fully
#     supported (explicitly, never silently) for reporting as a SEPARATE
#     diagnostic line — never presented as a second, equal-status official
#     baseline.
# Both names are defined ONCE here and reused everywhere else (this
# module's default arg, _VALID_MOMENTUM_COLS, and any caller) — never a
# redundant hardcoded "momentum_3m"/"momentum_6m" string anywhere else.
OFFICIAL_MOMENTUM_BASELINE_COL = "momentum_3m"
SECONDARY_MOMENTUM_DIAGNOSTIC_COL = "momentum_6m"
_VALID_MOMENTUM_COLS = (OFFICIAL_MOMENTUM_BASELINE_COL, SECONDARY_MOMENTUM_DIAGNOSTIC_COL)


def momentum_baseline(eval_rows: pd.DataFrame, momentum_col: str = OFFICIAL_MOMENTUM_BASELINE_COL) -> pd.Series:
    """Deterministic rule-based forecast: the formation-date's OWN trailing
    momentum feature value, used directly AS the forecast (ML_SPEC.md §15)
    — this is not a fitted regression around momentum, just the feature
    value itself, read from `eval_rows` (already purge-safe / formation-date
    feature values via `pipeline.folds.eval_rows_for_fold`). No fitting of
    any kind occurs in this function.

    `momentum_col` defaults to `OFFICIAL_MOMENTUM_BASELINE_COL`
    ("momentum_3m") — the LOCKED official Phase 4A momentum baseline.
    Passing `momentum_col=SECONDARY_MOMENTUM_DIAGNOSTIC_COL` ("momentum_6m")
    remains fully supported for explicit secondary-diagnostic reporting, but
    is never the default and must never be silently substituted for the
    official baseline. Any other value raises — no momentum lookback beyond
    these two named, locked columns is accepted.
    """
    if momentum_col not in _VALID_MOMENTUM_COLS:
        raise ValueError(
            f"momentum_baseline: momentum_col must be one of {_VALID_MOMENTUM_COLS} "
            f"(official = {OFFICIAL_MOMENTUM_BASELINE_COL!r}, secondary diagnostic = "
            f"{SECONDARY_MOMENTUM_DIAGNOSTIC_COL!r} — no other lookback is supported)."
        )
    return eval_rows.set_index(TICKER_COL)[momentum_col]


# ---------------------------------------------------------------------------
# C. Ridge / linear regression baseline
# ---------------------------------------------------------------------------
RIDGE_ALPHA_DEFAULT = 1.0  # Fixed Phase 4A default (task brief §9C, §12) — never
# selected using real OOS results; a future explicit tuning phase may revise
# this via internal temporal validation only (pipeline.folds.internal_validation_split).


def ridge_baseline(
    training_rows: pd.DataFrame,
    eval_rows: pd.DataFrame,
    feature_cols: Sequence[str],
    target_col: str = TARGET_COL,
    alpha: float = RIDGE_ALPHA_DEFAULT,
) -> Tuple[pd.Series, StandardScaler, Ridge]:
    """Ridge regression over `feature_cols`, with a `StandardScaler` fit
    STRICTLY on `training_rows` (ML_SPEC.md §22: "fit only on data available
    as of the relevant training window... never on the internal validation
    slice, the historical OOS evaluation period... or March"). `eval_rows`
    is only ever `.transform()`-ed, never seen by `.fit()` — see
    tests/test_baselines.py::test_ridge_scaler_never_fit_on_eval_rows for the
    adversarial proof (perturbing eval_rows must not change the fitted
    scaler's mean_/scale_).

    Returns (forecast Series indexed by ticker, the fitted scaler, the
    fitted model) so a caller/test can independently inspect the fitted
    preprocessing state without re-deriving it.
    """
    scaler = StandardScaler()
    X_train = scaler.fit_transform(training_rows[list(feature_cols)].to_numpy(dtype=float))
    y_train = training_rows[target_col].to_numpy(dtype=float)

    model = Ridge(alpha=alpha)
    model.fit(X_train, y_train)

    X_eval = scaler.transform(eval_rows[list(feature_cols)].to_numpy(dtype=float))
    preds = model.predict(X_eval)
    forecast = pd.Series(preds, index=eval_rows[TICKER_COL].to_numpy())
    return forecast, scaler, model
