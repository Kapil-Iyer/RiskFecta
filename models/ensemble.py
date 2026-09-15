"""
RiskFecta Phase 6A — locked equal-weight ensemble + model-comparison
diagnostics (ML_SPEC.md §19-§20; BUILD_PLAN.md Phase 6).

Consumes ONLY the already-frozen `predictions` rows written by Phase 4A
(`xgb_pred`) and Phase 5B (`lstm_pred`) — never fits, trains, or invokes
any model (`models/xgboost_model.py`, `models/lstm.py` are never imported
here). Identity/alignment is trivial by construction: xgb_pred and
lstm_pred already live on the SAME (ticker, forecast_date) row (both
model branches upsert into the same `predictions` table keyed on the
table's own UNIQUE(ticker, forecast_date) constraint — see
`pipeline/predictions.py`), so there is no join to get wrong; this module
reads that single canonical row per identity rather than rebuilding
predictions from model artifacts (RiskFecta 2.0 Phase 6A task brief §5).

Ensemble formula (ML_SPEC.md §19, LOCKED):

    ensemble_pred(i, t) = 0.5 * xgb_pred(i, t) + 0.5 * lstm_pred(i, t)

Exact arithmetic mean. No clipping, no normalization, no rank averaging,
no sign voting, no volatility/performance/ticker/date weighting. Not a
fitted model — no weight here is ever selected from OOS/validation
performance.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

from models.metrics import compute_all_metrics, pearson_corr, spearman_corr
from pipeline.sessions import TICKER_COL

FORECAST_DATE_COL = "forecast_date"
TARGET_DATE_COL = "target_date"
XGB_PRED_COL = "xgb_pred"
LSTM_PRED_COL = "lstm_pred"
ENSEMBLE_PRED_COL = "ensemble_pred"
ACTUAL_RETURN_COL = "actual_return"

# Canonical identity (task brief §5): read xgb_pred/lstm_pred/actual_return
# straight off the single predictions row per (ticker, forecast_date) —
# no inner join (would silently drop mismatches), no outer join (would
# admit rows missing a model prediction), no duplicate identity, no date
# shifting, no ticker remapping.
CANONICAL_IDENTITY_SQL = """
SELECT ticker, forecast_date, target_date, xgb_pred, lstm_pred, actual_return, directional_correct
FROM predictions
ORDER BY forecast_date, ticker
"""


def load_aligned_predictions(conn) -> pd.DataFrame:
    """Read-only load of the canonical (ticker, forecast_date) identity
    from `predictions`. Never writes anything."""
    df = pd.read_sql(CANONICAL_IDENTITY_SQL, conn)
    df[FORECAST_DATE_COL] = pd.to_datetime(df[FORECAST_DATE_COL])
    df[TARGET_DATE_COL] = pd.to_datetime(df[TARGET_DATE_COL])
    return df


def assert_identity_integrity(df: pd.DataFrame) -> None:
    """Fail loudly — never silently drop or average around the problem —
    on either integrity violation the task brief calls out explicitly:

      D. a row missing either model's prediction (no silent inner-join
         drop, no outer-join fill).
      E. a duplicate (ticker, forecast_date) identity.
    """
    dup_mask = df.duplicated(subset=[TICKER_COL, FORECAST_DATE_COL], keep=False)
    if dup_mask.any():
        n = int(df.duplicated(subset=[TICKER_COL, FORECAST_DATE_COL]).sum())
        raise ValueError(
            f"assert_identity_integrity: {n} duplicate (ticker, forecast_date) identities found"
        )
    missing_xgb = df[XGB_PRED_COL].isna()
    missing_lstm = df[LSTM_PRED_COL].isna()
    if missing_xgb.any() or missing_lstm.any():
        raise ValueError(
            "assert_identity_integrity: missing model prediction(s) for the ensemble — "
            f"{int(missing_xgb.sum())} row(s) missing xgb_pred, "
            f"{int(missing_lstm.sum())} row(s) missing lstm_pred. "
            "Every aligned row must carry both models' predictions before an "
            "ensemble value can be computed for it."
        )


def assert_expected_shape(df: pd.DataFrame, n_tickers: int = 50, n_dates: int = 47) -> None:
    """Defensive shape check against the Phase 6A pre-execution live
    invariant (task brief §3, §14): exactly n_tickers * n_dates aligned
    rows. Not called by every test (synthetic fixtures are smaller by
    design) — only by the real-data smoke check."""
    expected = n_tickers * n_dates
    if len(df) != expected:
        raise ValueError(
            f"assert_expected_shape: expected {expected} rows ({n_tickers} tickers x {n_dates} dates), got {len(df)}"
        )
    if df[TICKER_COL].nunique() != n_tickers:
        raise ValueError(f"assert_expected_shape: expected {n_tickers} distinct tickers, got {df[TICKER_COL].nunique()}")
    if df[FORECAST_DATE_COL].nunique() != n_dates:
        raise ValueError(f"assert_expected_shape: expected {n_dates} distinct forecast dates, got {df[FORECAST_DATE_COL].nunique()}")


# ---------------------------------------------------------------------------
# Locked ensemble arithmetic
# ---------------------------------------------------------------------------
def compute_ensemble_pred(xgb_pred: pd.Series, lstm_pred: pd.Series) -> pd.Series:
    """Exact 50/50 arithmetic mean (ML_SPEC.md §19). No clipping, no
    normalization, no transformation of any kind. NaN propagates through
    the arithmetic rather than being silently treated as zero."""
    return 0.5 * xgb_pred + 0.5 * lstm_pred


def build_ensemble_frame(df: pd.DataFrame) -> pd.DataFrame:
    """`df` must carry ticker/forecast_date/xgb_pred/lstm_pred (e.g.
    `load_aligned_predictions` output). Validates identity integrity first
    (fail loudly, never silently drop/fill), then adds `ensemble_pred`.
    """
    assert_identity_integrity(df)
    out = df.copy()
    out[ENSEMBLE_PRED_COL] = compute_ensemble_pred(out[XGB_PRED_COL], out[LSTM_PRED_COL])
    return out


# ---------------------------------------------------------------------------
# Diversity / complementarity diagnostics (task brief §8)
# ---------------------------------------------------------------------------
@dataclass
class CorrelationResult:
    pearson: float
    spearman: float


def prediction_correlation(xgb_pred: Sequence[float], lstm_pred: Sequence[float]) -> CorrelationResult:
    """A. corr(xgb_pred, lstm_pred). B. rank_corr(xgb_pred, lstm_pred)."""
    return CorrelationResult(
        pearson=pearson_corr(xgb_pred, lstm_pred),
        spearman=spearman_corr(xgb_pred, lstm_pred),
    )


def residual_correlation(
    xgb_pred: Sequence[float], lstm_pred: Sequence[float], actual_return: Sequence[float]
) -> CorrelationResult:
    """C. residual/error correlation, residual = prediction - actual_return."""
    x = np.asarray(xgb_pred, dtype=float)
    l = np.asarray(lstm_pred, dtype=float)
    a = np.asarray(actual_return, dtype=float)
    xgb_resid = x - a
    lstm_resid = l - a
    return CorrelationResult(
        pearson=pearson_corr(xgb_resid, lstm_resid),
        spearman=spearman_corr(xgb_resid, lstm_resid),
    )


@dataclass
class DirectionalAgreementResult:
    n_total: int
    n_zero_sign: int      # either model predicted exactly 0 -> excluded (undefined agreement)
    n_comparable: int     # both signs non-zero -> agreement is defined
    n_agree: int
    n_disagree: int
    agreement_rate: float  # n_agree / n_comparable; NaN if n_comparable == 0


def directional_agreement(xgb_pred: Sequence[float], lstm_pred: Sequence[float]) -> DirectionalAgreementResult:
    """D. fraction where sign(xgb_pred) == sign(lstm_pred), with EXPLICIT
    zero-sign handling: a row where either model predicts exactly 0 has no
    defined direction to agree/disagree on, so it is excluded from the
    comparable denominator rather than silently counted either way (same
    convention as `models.metrics.directional_accuracy`'s own zero-actual
    exclusion).
    E. disagreement count/rate is `n_disagree` / `n_comparable`.
    """
    x = np.asarray(xgb_pred, dtype=float)
    l = np.asarray(lstm_pred, dtype=float)
    sx, sl = np.sign(x), np.sign(l)
    zero_sign = (sx == 0) | (sl == 0)
    comparable = ~zero_sign
    n_comparable = int(comparable.sum())
    agree = comparable & (sx == sl)
    n_agree = int(agree.sum())
    n_disagree = n_comparable - n_agree
    rate = float(n_agree / n_comparable) if n_comparable > 0 else float("nan")
    return DirectionalAgreementResult(
        n_total=len(x),
        n_zero_sign=int(zero_sign.sum()),
        n_comparable=n_comparable,
        n_agree=n_agree,
        n_disagree=n_disagree,
        agreement_rate=rate,
    )


@dataclass
class DisagreementBreakdown:
    n_disagree: int
    xgb_correct_lstm_wrong: int
    lstm_correct_xgb_wrong: int
    actual_zero_excluded: int  # models disagree, but realized return has no sign to match either way


def disagreement_outcome_breakdown(
    xgb_pred: Sequence[float], lstm_pred: Sequence[float], actual_return: Sequence[float]
) -> DisagreementBreakdown:
    """F. Among directionally-disagreeing rows (both signs non-zero and
    opposite), which model called the realized direction correctly.
    Because a disagreement row has sx == -sl (both nonzero), and a nonzero
    realized sign sa can equal at most one of {sx, sl}, every such row
    falls into exactly one of: xgb-correct, lstm-correct, or
    actual-return-exactly-zero (no sign to match — the explicit zero edge
    case, same convention as `models.metrics.directional_accuracy`).
    """
    x = np.asarray(xgb_pred, dtype=float)
    l = np.asarray(lstm_pred, dtype=float)
    a = np.asarray(actual_return, dtype=float)
    sx, sl, sa = np.sign(x), np.sign(l), np.sign(a)

    disagree = (sx != 0) & (sl != 0) & (sx != sl)
    zero_actual = disagree & (sa == 0)
    scored = disagree & (sa != 0)
    xgb_correct = scored & (sx == sa)
    lstm_correct = scored & (sl == sa)

    return DisagreementBreakdown(
        n_disagree=int(disagree.sum()),
        xgb_correct_lstm_wrong=int(xgb_correct.sum()),
        lstm_correct_xgb_wrong=int(lstm_correct.sum()),
        actual_zero_excluded=int(zero_actual.sum()),
    )


@dataclass
class DispersionDiagnostics:
    var_xgb: float
    var_lstm: float
    var_ensemble: float
    mean_abs_diff_xgb_lstm: float  # direct magnitude-of-disagreement measure


def dispersion_diagnostics(
    xgb_pred: Sequence[float], lstm_pred: Sequence[float], ensemble_pred: Sequence[float]
) -> DispersionDiagnostics:
    """G. ensemble variance / prediction dispersion vs. the individual
    branches."""
    x = np.asarray(xgb_pred, dtype=float)
    l = np.asarray(lstm_pred, dtype=float)
    e = np.asarray(ensemble_pred, dtype=float)
    return DispersionDiagnostics(
        var_xgb=float(np.nanvar(x)),
        var_lstm=float(np.nanvar(l)),
        var_ensemble=float(np.nanvar(e)),
        mean_abs_diff_xgb_lstm=float(np.nanmean(np.abs(x - l))),
    )


# ---------------------------------------------------------------------------
# Cross-sectional usefulness (task brief §9; ML_SPEC.md §20)
# ---------------------------------------------------------------------------
def per_date_rank_ic(
    df: pd.DataFrame,
    pred_col: str,
    actual_col: str = ACTUAL_RETURN_COL,
    date_col: str = FORECAST_DATE_COL,
) -> pd.DataFrame:
    """Per-formation-date cross-sectional Spearman rank correlation
    between `pred_col` and realized `actual_col` (ML_SPEC.md §20:
    "forecast rank vs. realized rank, per formation date"). One row per
    date; `rank_ic` is NaN for a date with fewer than 2 valid
    (pred, actual) pairs or a constant vector — reuses
    `models.metrics.spearman_corr` verbatim, no metric redefinition.
    """
    rows = []
    for date, g in df.groupby(date_col):
        ic = spearman_corr(g[actual_col], g[pred_col])
        n = int(g[[pred_col, actual_col]].dropna().shape[0])
        rows.append({date_col: date, "rank_ic": ic, "n": n})
    return pd.DataFrame(rows, columns=[date_col, "rank_ic", "n"]).sort_values(date_col).reset_index(drop=True)


def rank_ic_summary(rank_ic_df: pd.DataFrame) -> dict:
    """Distribution / mean / median of per-date rank IC, and the fraction
    of dates with positive rank IC (task brief §9)."""
    ic = rank_ic_df["rank_ic"].dropna()
    if len(ic) == 0:
        return {"mean": float("nan"), "median": float("nan"), "frac_positive": float("nan"), "n_dates": 0}
    return {
        "mean": float(ic.mean()),
        "median": float(ic.median()),
        "frac_positive": float((ic > 0).mean()),
        "n_dates": int(len(ic)),
    }


# Quintile (10-of-50 names per group) top-vs-bottom diagnostic. ML_SPEC.md
# §20 names this construction with its own example units — "realized
# return spread between top- and bottom-ranked forecast deciles/
# quintiles... where statistically appropriate given the 50-name
# universe" — so this is the spec's own example, not an invented
# methodology: a decile group (5 of 50 names) is too thin for a 50-name
# cross-section, so quintile (10 of 50, 20% per group) is used.
DEFAULT_TOP_BOTTOM_GROUP_SIZE = 10


def top_bottom_quintile_spread(
    df: pd.DataFrame,
    pred_col: str,
    actual_col: str = ACTUAL_RETURN_COL,
    date_col: str = FORECAST_DATE_COL,
    group_size: int = DEFAULT_TOP_BOTTOM_GROUP_SIZE,
) -> pd.DataFrame:
    """Per-date realized-return spread between the top-`group_size` and
    bottom-`group_size` names ranked by `pred_col` (ML_SPEC.md §20). A
    date with fewer than 2 * group_size valid (pred, actual) rows yields
    NaN rather than a group built from an under-sized/overlapping pool.
    """
    rows = []
    for date, g in df.groupby(date_col):
        g = g.dropna(subset=[pred_col, actual_col])
        if len(g) < 2 * group_size:
            rows.append(
                {date_col: date, "top_mean_return": float("nan"), "bottom_mean_return": float("nan"),
                 "spread": float("nan"), "n": len(g)}
            )
            continue
        ranked = g.sort_values(pred_col, ascending=False)
        top = ranked.iloc[:group_size]
        bottom = ranked.iloc[-group_size:]
        top_mean = float(top[actual_col].mean())
        bottom_mean = float(bottom[actual_col].mean())
        rows.append(
            {date_col: date, "top_mean_return": top_mean, "bottom_mean_return": bottom_mean,
             "spread": top_mean - bottom_mean, "n": len(g)}
        )
    return pd.DataFrame(
        rows, columns=[date_col, "top_mean_return", "bottom_mean_return", "spread", "n"]
    ).sort_values(date_col).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Model comparison (task brief §7)
# ---------------------------------------------------------------------------
DEFAULT_MODEL_COLS = (XGB_PRED_COL, LSTM_PRED_COL, ENSEMBLE_PRED_COL)


def compare_models(
    df: pd.DataFrame,
    model_cols: Sequence[str] = DEFAULT_MODEL_COLS,
    actual_col: str = ACTUAL_RETURN_COL,
) -> pd.DataFrame:
    """ML_SPEC.md §20 metrics (MAE, RMSE, directional accuracy, Pearson,
    Spearman) for each column in `model_cols` against `actual_col`,
    reusing `models.metrics.compute_all_metrics` verbatim — no metric is
    redefined here.

    Historical-mean/momentum/Ridge BASELINE columns are intentionally not
    included in `DEFAULT_MODEL_COLS`: their OOS forecasts are not
    persisted anywhere in the frozen schema (see
    `pipeline/predictions.py` module docstring — baselines are
    evaluation-only and were never written to `predictions`), so they are
    not a frozen input this phase can consume without recomputing them.
    This is reported as an unresolved methodology ambiguity rather than
    silently invented (task brief §9, §10, §20-14) — a caller may still
    pass baseline prediction columns explicitly via `model_cols` if a
    future phase makes them available.
    """
    rows = []
    for col in model_cols:
        if col not in df.columns:
            continue
        m = compute_all_metrics(df[actual_col], df[col])
        m["model"] = col
        rows.append(m)
    return pd.DataFrame(rows).set_index("model") if rows else pd.DataFrame()
