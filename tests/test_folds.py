"""
RiskFecta Phase 4A — walk-forward fold construction tests (pipeline/folds.py).

Covers task brief §3B, §17 (A/B/C/D/E/F/M), §24 (formation/label boundary
worked example). Every test here proves fold STRUCTURE and row SELECTION
without ever fitting a model (task brief §4).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pipeline.folds import (
    Fold,
    attach_global_session,
    build_folds,
    build_global_calendar,
    eval_rows_for_fold,
    fold_for_formation_session,
    internal_validation_split,
    rows_in_session_range,
    training_rows_for_fold,
)
from pipeline.targets import TARGET_COL

FEATURE_COLS = ["vix", "yield_10y", "momentum_3m", "momentum_6m", "volatility_20d"]


def _prices(ticker: str, n: int, start: str = "2020-01-01") -> pd.DataFrame:
    dates = pd.bdate_range(start, periods=n)
    return pd.DataFrame({"ticker": ticker, "date": dates, "close": 100.0 + np.arange(n)})


def _panel(tickers, n_sessions: int, feature_fn=None, target_fn=None) -> pd.DataFrame:
    """Build a synthetic per-(ticker, date) panel with global_session-ready
    columns: ticker, date, feature_cols..., target_21d. feature_fn/target_fn
    map (ticker, session_idx) -> float; default to deterministic values that
    make it trivial to assert exactly which rows were selected.
    """
    dates = pd.bdate_range("2020-01-01", periods=n_sessions)
    rows = []
    for ticker in tickers:
        for s, d in enumerate(dates):
            row = {"ticker": ticker, "date": d}
            for col in FEATURE_COLS:
                row[col] = (feature_fn(ticker, s, col) if feature_fn else float(s))
            row[TARGET_COL] = (target_fn(ticker, s) if target_fn else float(s) / 1000.0)
            rows.append(row)
    return pd.DataFrame(rows)


def _prices_from_panel(panel: pd.DataFrame) -> pd.DataFrame:
    out = panel[["ticker", "date"]].copy()
    out["close"] = 100.0
    return out


# ---------------------------------------------------------------------------
# §24 — Formation/label boundary worked numeric example (T = 300)
# ---------------------------------------------------------------------------
def test_formation_label_boundary_worked_example_T_300():
    calendar = pd.bdate_range("2015-01-01", periods=400)
    fold = fold_for_formation_session(T=300, calendar=calendar, train_window=252, horizon=21)

    assert fold.formation_session == 300
    assert fold.train_window_first_session == 49          # T - 252 + 1
    assert fold.train_window_last_session == 300           # == T
    assert fold.label_eligible_last_session == 279          # T - 21
    assert fold.eval_session == 300
    assert fold.target_realized_session == 321              # T + 21

    # Last eligible training feature session: 279. Its target endpoint:
    last_eligible = fold.label_eligible_last_session
    endpoint = last_eligible + 21
    assert endpoint == 300
    assert endpoint <= fold.formation_session  # ALLOWED (endpoint == T is allowed)

    # First PURGED session: 280. Its target endpoint:
    first_purged = fold.label_eligible_last_session + 1
    purged_endpoint = first_purged + 21
    assert purged_endpoint == 301
    assert purged_endpoint > fold.formation_session  # PURGED

    # Usable labeled sessions per ticker before feature-availability filtering:
    # 279 - 49 + 1 = 231 = TRAIN_WINDOW - FORECAST_HORIZON = 252 - 21.
    assert fold.usable_labeled_sessions_before_feature_filtering == 231
    assert fold.usable_labeled_sessions_before_feature_filtering == 252 - 21


# ---------------------------------------------------------------------------
# A. LABEL AVAILABILITY — exact boundary, one-before, one-after
# ---------------------------------------------------------------------------
def test_label_availability_one_before_boundary_allowed():
    """A row whose target endpoint is clearly BEFORE the cutoff is allowed."""
    calendar = pd.bdate_range("2015-01-01", periods=60)
    fold = fold_for_formation_session(T=50, calendar=calendar, train_window=40, horizon=10)
    # session 30: endpoint = 40, cutoff T=50 -> clearly before, allowed.
    assert fold.train_window_first_session <= 30 <= fold.label_eligible_last_session


def test_label_availability_exact_boundary_allowed():
    """endpoint == T is ALLOWED under the resolved formation-cutoff semantics."""
    calendar = pd.bdate_range("2015-01-01", periods=60)
    fold = fold_for_formation_session(T=50, calendar=calendar, train_window=40, horizon=10)
    s = fold.label_eligible_last_session  # endpoint = s + 10 == 50 == T
    assert s + 10 == fold.formation_session
    assert s <= fold.label_eligible_last_session  # included in the eligible range


def test_label_availability_one_after_boundary_rejected():
    calendar = pd.bdate_range("2015-01-01", periods=60)
    fold = fold_for_formation_session(T=50, calendar=calendar, train_window=40, horizon=10)
    s = fold.label_eligible_last_session + 1  # endpoint = 51 > T=50
    assert s + 10 > fold.formation_session
    assert s > fold.label_eligible_last_session  # excluded


def test_no_training_label_endpoint_exceeds_formation_session_across_all_folds():
    """No training label endpoint ever enters/exceeds the OOS formation
    session, checked structurally across an entire fold sequence."""
    calendar = pd.bdate_range("2015-01-01", periods=500)
    folds = build_folds(calendar, train_window=100, step=21, horizon=21)
    assert len(folds) > 1
    for fold in folds:
        max_endpoint = fold.label_eligible_last_session + 21
        assert max_endpoint <= fold.formation_session


def test_purge_is_session_based_not_calendar_based():
    """Insert a large calendar gap (simulating an irregular trading
    calendar) and confirm the purge boundary is measured in valid-session
    COUNT, not elapsed calendar days."""
    # 60 consecutive business days, then a 90-calendar-day gap, then 60 more.
    first = pd.bdate_range("2015-01-01", periods=60)
    second = pd.bdate_range(first[-1] + pd.Timedelta(days=90), periods=60)
    calendar = pd.DatetimeIndex(list(first) + list(second))

    fold = fold_for_formation_session(T=100, calendar=calendar, train_window=40, horizon=10)
    # Purely index-arithmetic — the gap in *dates* must not change the
    # session-count-based boundary at all.
    assert fold.label_eligible_last_session == 90  # 100 - 10
    assert fold.train_window_first_session == 61    # 100 - 40 + 1
    # And the *date* span across the gap is much larger than 40 calendar days,
    # proving the window is genuinely session-counted, not date-subtracted.
    span_days = (calendar[fold.train_window_last_session] - calendar[fold.train_window_first_session]).days
    assert span_days > 40


# ---------------------------------------------------------------------------
# B. TRAIN WINDOW — rolling 252-session candidate window, exact start/end
# ---------------------------------------------------------------------------
def test_train_window_exact_span_252_sessions():
    calendar = pd.bdate_range("2015-01-01", periods=400)
    fold = fold_for_formation_session(T=300, calendar=calendar, train_window=252, horizon=21)
    span = fold.train_window_last_session - fold.train_window_first_session + 1
    assert span == 252


def test_build_folds_first_formation_session_requires_full_window():
    calendar = pd.bdate_range("2015-01-01", periods=400)
    folds = build_folds(calendar, train_window=252, step=21, horizon=21)
    assert folds[0].formation_session == 251  # train_window - 1
    assert folds[0].train_window_first_session == 0


# ---------------------------------------------------------------------------
# C. STEP — exactly 21 valid sessions, not calendar days
# ---------------------------------------------------------------------------
def test_folds_advance_by_exactly_step_sessions():
    calendar = pd.bdate_range("2015-01-01", periods=500)
    folds = build_folds(calendar, train_window=100, step=21, horizon=21)
    diffs = [b.formation_session - a.formation_session for a, b in zip(folds, folds[1:])]
    assert all(d == 21 for d in diffs)


def test_step_is_session_indexed_survives_calendar_gap():
    first = pd.bdate_range("2015-01-01", periods=150)
    second = pd.bdate_range(first[-1] + pd.Timedelta(days=60), periods=150)
    calendar = pd.DatetimeIndex(list(first) + list(second))
    folds = build_folds(calendar, train_window=100, step=21, horizon=21)
    diffs = [b.formation_session - a.formation_session for a, b in zip(folds, folds[1:])]
    assert all(d == 21 for d in diffs)  # session count, unaffected by the date gap


# ---------------------------------------------------------------------------
# D. OOS SEPARATION — train/eval disjoint, no OOS rows used in fitting
# ---------------------------------------------------------------------------
def test_train_and_eval_sessions_are_disjoint():
    calendar = pd.bdate_range("2015-01-01", periods=400)
    fold = fold_for_formation_session(T=300, calendar=calendar, train_window=252, horizon=21)
    assert fold.eval_session > fold.label_eligible_last_session


def test_eval_rows_never_appear_in_training_rows_selection():
    panel = _panel(["AAPL", "MSFT"], n_sessions=400)
    panel = attach_global_session(panel, pd.bdate_range("2020-01-01", periods=400))
    fold = fold_for_formation_session(T=300, calendar=pd.bdate_range("2020-01-01", periods=400),
                                       train_window=252, horizon=21)

    train_rows = training_rows_for_fold(panel, fold, FEATURE_COLS)
    eval_rows = eval_rows_for_fold(panel, fold, FEATURE_COLS)

    train_keys = set(zip(train_rows["ticker"], train_rows["date"]))
    eval_keys = set(zip(eval_rows["ticker"], eval_rows["date"]))
    assert train_keys.isdisjoint(eval_keys)
    assert (train_rows["global_session"] <= fold.label_eligible_last_session).all()
    assert (eval_rows["global_session"] == fold.eval_session).all()


# ---------------------------------------------------------------------------
# E. POOLED SEMANTICS — all tickers combined for training, identity retained
# ---------------------------------------------------------------------------
def test_training_rows_pool_across_all_tickers():
    tickers = ["AAPL", "MSFT", "JPM"]
    panel = _panel(tickers, n_sessions=400)
    calendar = pd.bdate_range("2020-01-01", periods=400)
    panel = attach_global_session(panel, calendar)
    fold = fold_for_formation_session(T=300, calendar=calendar, train_window=252, horizon=21)

    train_rows = training_rows_for_fold(panel, fold, FEATURE_COLS)
    assert set(train_rows["ticker"].unique()) == set(tickers)


def test_eval_rows_retain_ticker_and_date_identity():
    tickers = ["AAPL", "MSFT"]
    panel = _panel(tickers, n_sessions=400)
    calendar = pd.bdate_range("2020-01-01", periods=400)
    panel = attach_global_session(panel, calendar)
    fold = fold_for_formation_session(T=300, calendar=calendar, train_window=252, horizon=21)

    eval_rows = eval_rows_for_fold(panel, fold, FEATURE_COLS)
    assert set(eval_rows["ticker"]) == set(tickers)
    assert (eval_rows["date"] == calendar[300]).all()


# ---------------------------------------------------------------------------
# F. TICKER ISOLATION — targets/features never cross ticker boundaries
# ---------------------------------------------------------------------------
def test_ticker_with_missing_session_does_not_affect_other_tickers_fold_rows():
    calendar = pd.bdate_range("2020-01-01", periods=400)
    panel = _panel(["AAPL", "MSFT"], n_sessions=400)
    panel = attach_global_session(panel, calendar)

    # AAPL is missing its formation-date row entirely (e.g. a genuine gap).
    fold = fold_for_formation_session(T=300, calendar=calendar, train_window=252, horizon=21)
    aapl_eval_date = panel.loc[(panel["ticker"] == "AAPL") & (panel["global_session"] == 300)].index
    panel_missing = panel.drop(index=aapl_eval_date)

    eval_rows = eval_rows_for_fold(panel_missing, fold, FEATURE_COLS)
    # AAPL absent, MSFT unaffected.
    assert "AAPL" not in set(eval_rows["ticker"])
    assert "MSFT" in set(eval_rows["ticker"])
    msft_row = eval_rows[eval_rows["ticker"] == "MSFT"]
    assert len(msft_row) == 1


def test_pooled_interleaved_frame_purge_matches_per_ticker_purge():
    """Purge correctness must be identical whether tickers are interleaved
    in one pooled frame or processed one at a time (task brief §3B.7)."""
    calendar = pd.bdate_range("2020-01-01", periods=400)
    fold = fold_for_formation_session(T=300, calendar=calendar, train_window=252, horizon=21)

    aapl = _panel(["AAPL"], n_sessions=400)
    msft = _panel(["MSFT"], n_sessions=400)
    pooled = pd.concat([aapl, msft], ignore_index=True).sample(frac=1.0, random_state=3).reset_index(drop=True)

    aapl_g = attach_global_session(aapl, calendar)
    msft_g = attach_global_session(msft, calendar)
    pooled_g = attach_global_session(pooled, calendar)

    train_aapl_solo = training_rows_for_fold(aapl_g, fold, FEATURE_COLS)
    train_msft_solo = training_rows_for_fold(msft_g, fold, FEATURE_COLS)
    train_pooled = training_rows_for_fold(pooled_g, fold, FEATURE_COLS)

    assert len(train_pooled) == len(train_aapl_solo) + len(train_msft_solo)
    assert set(train_pooled.loc[train_pooled["ticker"] == "AAPL", "date"]) == set(train_aapl_solo["date"])
    assert set(train_pooled.loc[train_pooled["ticker"] == "MSFT", "date"]) == set(train_msft_solo["date"])


# ---------------------------------------------------------------------------
# Missing-feature handling (task brief §8) — no zero-fill/bfill, rows dropped
# ---------------------------------------------------------------------------
def test_rows_with_missing_required_feature_are_dropped_not_filled():
    calendar = pd.bdate_range("2020-01-01", periods=400)
    panel = _panel(["AAPL"], n_sessions=400)
    panel = attach_global_session(panel, calendar)
    # Blank out vix for one eligible training session.
    panel.loc[(panel["ticker"] == "AAPL") & (panel["global_session"] == 200), "vix"] = np.nan

    fold = fold_for_formation_session(T=300, calendar=calendar, train_window=252, horizon=21)
    train_rows = training_rows_for_fold(panel, fold, FEATURE_COLS)
    assert 200 not in set(train_rows["global_session"])
    # No fabricated value anywhere in the surviving rows.
    assert train_rows[FEATURE_COLS].isna().sum().sum() == 0


def test_rows_with_missing_target_are_dropped():
    calendar = pd.bdate_range("2020-01-01", periods=400)
    panel = _panel(["AAPL"], n_sessions=400)
    panel = attach_global_session(panel, calendar)
    panel.loc[(panel["ticker"] == "AAPL") & (panel["global_session"] == 200), TARGET_COL] = np.nan

    fold = fold_for_formation_session(T=300, calendar=calendar, train_window=252, horizon=21)
    train_rows = training_rows_for_fold(panel, fold, FEATURE_COLS)
    assert 200 not in set(train_rows["global_session"])


# ---------------------------------------------------------------------------
# M. FUTURE PERTURBATION — on the FOLD BUILDER itself (not just features)
# ---------------------------------------------------------------------------
def test_future_perturbation_does_not_change_earlier_fold_training_selection():
    """Perturbing future features/targets/prices must not change an
    EARLIER fold's: eligible training rows, their feature/target values, or
    the eval-row selection. This is the fold-builder-level causality test
    the user explicitly asked for (distinct from pipeline/targets.py's own
    feature/target-level perturbation test)."""
    calendar = pd.bdate_range("2020-01-01", periods=500)
    tickers = ["AAPL", "MSFT", "JPM"]
    panel = _panel(tickers, n_sessions=500)
    panel = attach_global_session(panel, calendar)

    earlier_fold = fold_for_formation_session(T=300, calendar=calendar, train_window=252, horizon=21)
    later_fold = fold_for_formation_session(T=450, calendar=calendar, train_window=252, horizon=21)

    before_train = training_rows_for_fold(panel, earlier_fold, FEATURE_COLS).sort_values(["ticker", "date"]).reset_index(drop=True)
    before_eval = eval_rows_for_fold(panel, earlier_fold, FEATURE_COLS).sort_values(["ticker", "date"]).reset_index(drop=True)

    # Perturb everything strictly AFTER the earlier fold's own formation
    # session (300): features, target, and an injected extra column, at a
    # session that belongs to the LATER fold's training/eval window.
    perturbed = panel.copy()
    future_mask = perturbed["global_session"] > earlier_fold.formation_session
    for col in FEATURE_COLS:
        perturbed.loc[future_mask, col] = perturbed.loc[future_mask, col] * 1000.0 + 999.0
    perturbed.loc[future_mask, TARGET_COL] = perturbed.loc[future_mask, TARGET_COL] * -1000.0

    after_train = training_rows_for_fold(perturbed, earlier_fold, FEATURE_COLS).sort_values(["ticker", "date"]).reset_index(drop=True)
    after_eval = eval_rows_for_fold(perturbed, earlier_fold, FEATURE_COLS).sort_values(["ticker", "date"]).reset_index(drop=True)

    pd.testing.assert_frame_equal(before_train, after_train)
    pd.testing.assert_frame_equal(before_eval, after_eval)

    # Sanity: the later fold's own selection DOES change (proves the
    # perturbation was real and reachable, not a no-op).
    later_before = training_rows_for_fold(panel, later_fold, FEATURE_COLS)
    later_after = training_rows_for_fold(perturbed, later_fold, FEATURE_COLS)
    assert not later_before.equals(later_after)


def test_future_perturbation_of_a_ticker_never_affects_another_tickers_rows():
    calendar = pd.bdate_range("2020-01-01", periods=500)
    panel = _panel(["AAPL", "MSFT"], n_sessions=500)
    panel = attach_global_session(panel, calendar)
    fold = fold_for_formation_session(T=300, calendar=calendar, train_window=252, horizon=21)

    before_msft = training_rows_for_fold(panel, fold, FEATURE_COLS)
    before_msft = before_msft[before_msft["ticker"] == "MSFT"].sort_values("date").reset_index(drop=True)

    perturbed = panel.copy()
    aapl_mask = perturbed["ticker"] == "AAPL"
    perturbed.loc[aapl_mask, "vix"] = perturbed.loc[aapl_mask, "vix"] * -999.0
    perturbed.loc[aapl_mask, TARGET_COL] = perturbed.loc[aapl_mask, TARGET_COL] * -999.0

    after_msft = training_rows_for_fold(perturbed, fold, FEATURE_COLS)
    after_msft = after_msft[after_msft["ticker"] == "MSFT"].sort_values("date").reset_index(drop=True)

    pd.testing.assert_frame_equal(before_msft, after_msft)


# ---------------------------------------------------------------------------
# Global calendar / session attachment
# ---------------------------------------------------------------------------
def test_build_global_calendar_is_sorted_deduplicated_union():
    aapl = _prices("AAPL", 5, start="2020-01-01")
    msft = _prices("MSFT", 5, start="2020-01-03")  # overlaps + extends
    prices = pd.concat([aapl, msft], ignore_index=True)
    calendar = build_global_calendar(prices)
    assert calendar.is_monotonic_increasing
    assert calendar.is_unique
    assert len(calendar) == len(set(aapl["date"]) | set(msft["date"]))


def test_attach_global_session_gives_nan_for_ticker_gap_not_a_fabricated_index():
    calendar = pd.bdate_range("2020-01-01", periods=10)
    df = pd.DataFrame({"ticker": "AAPL", "date": [calendar[0], calendar[3], calendar[9]]})
    out = attach_global_session(df, calendar)
    assert out["global_session"].tolist() == [0, 3, 9]

    # A date NOT in the calendar at all must map to NaN, never a nearest index.
    stray = pd.DataFrame({"ticker": "AAPL", "date": [pd.Timestamp("2020-06-01")]})
    out2 = attach_global_session(stray, calendar)
    assert out2["global_session"].isna().all()


# ---------------------------------------------------------------------------
# J. Internal temporal validation — temporally later, still purged, outer-OOS blind
# ---------------------------------------------------------------------------
def test_internal_validation_split_is_temporal_and_outer_oos_blind():
    calendar = pd.bdate_range("2020-01-01", periods=400)
    fold = fold_for_formation_session(T=300, calendar=calendar, train_window=252, horizon=21)

    inner_train, inner_val = internal_validation_split(fold, val_sessions=21)

    # Strictly temporal, non-overlapping.
    assert inner_train[1] < inner_val[0]
    # Both ranges are within the fold's own already-purged eligible range.
    assert fold.train_window_first_session <= inner_train[0]
    assert inner_val[1] <= fold.label_eligible_last_session
    # Outer-OOS-blind: inner_val never reaches the formation/eval session.
    assert inner_val[1] < fold.eval_session


def test_internal_validation_rows_respect_same_purge_as_outer_fold():
    calendar = pd.bdate_range("2020-01-01", periods=400)
    panel = _panel(["AAPL", "MSFT"], n_sessions=400)
    panel = attach_global_session(panel, calendar)
    fold = fold_for_formation_session(T=300, calendar=calendar, train_window=252, horizon=21)

    inner_train_range, inner_val_range = internal_validation_split(fold, val_sessions=21)
    inner_train_rows = rows_in_session_range(panel, inner_train_range, FEATURE_COLS)
    inner_val_rows = rows_in_session_range(panel, inner_val_range, FEATURE_COLS)

    assert inner_train_rows["global_session"].max() < inner_val_rows["global_session"].min()
    assert inner_val_rows["global_session"].max() <= fold.label_eligible_last_session
    assert inner_val_rows["global_session"].max() < fold.eval_session


def test_internal_validation_split_raises_when_val_sessions_too_large():
    calendar = pd.bdate_range("2020-01-01", periods=100)
    fold = fold_for_formation_session(T=60, calendar=calendar, train_window=40, horizon=10)
    with pytest.raises(ValueError):
        internal_validation_split(fold, val_sessions=1000)


def test_build_folds_rejects_train_window_not_exceeding_horizon():
    calendar = pd.bdate_range("2020-01-01", periods=100)
    with pytest.raises(ValueError):
        build_folds(calendar, train_window=10, step=5, horizon=21)
