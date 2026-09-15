"""
RiskFecta Phase 5A — LSTM fold-wiring delta tests (pipeline/lstm_folds.py).

Synthetic only (task brief §7-§9 of "Continue RiskFecta 2.0 Phase 5A").
Proves Reading (A) — TRAIN_WINDOW=252 bounds each sample's own END
session only, never its causal input reach — is implemented exactly as
locked, with no fabricated pre-history and no outer-OOS leakage.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import config
from pipeline.folds import GLOBAL_SESSION_COL, fold_for_formation_session
from pipeline.lstm_folds import build_lstm_fold_sequences
from pipeline.sequences import (
    LSTM_FEATURE_COLS,
    SEQ_END_POS_COL,
    SEQ_START_POS_COL,
    SEQ_TICKER_COL,
    build_sequence_index,
    extract_sequence_array,
)
from pipeline.targets import TARGET_COL


def _make_panel(tickers, n_sessions, start="2020-01-06", target_horizon=None):
    """Full-history synthetic panel (no warm-up NaNs) — every ticker has
    genuine complete observations from session 0 through n_sessions-1, so
    "does 60-session pre-window history exist" can be answered honestly
    for any candidate end session. Feature value at (ticker t_i, position
    p, column j) = t_i*10000 + p*10 + j, exactly reconstructable."""
    target_horizon = config.FORECAST_HORIZON if target_horizon is None else target_horizon
    dates = pd.bdate_range(start=start, periods=n_sessions)
    frames = []
    for t_i, ticker in enumerate(tickers):
        pos = np.arange(n_sessions)
        df = pd.DataFrame({"ticker": ticker, "date": dates})
        for j, col in enumerate(LSTM_FEATURE_COLS):
            df[col] = t_i * 10000 + pos * 10 + j
        tri = 100.0 + pos * 0.05
        target = pd.Series(tri).shift(-target_horizon) / pd.Series(tri) - 1
        df[TARGET_COL] = target.to_numpy()
        frames.append(df)
    panel = pd.concat(frames, ignore_index=True)
    return panel.sort_values(["ticker", "date"], kind="mergesort").reset_index(drop=True)


TICKERS = ["AAA", "BBB"]
N_SESSIONS = 350  # >= T(300) + horizon(21) + buffer, so OOS targets are realizable and pre-history is abundant


@pytest.fixture(scope="module")
def panel_and_calendar():
    panel = _make_panel(TICKERS, N_SESSIONS)
    calendar = pd.DatetimeIndex(sorted(panel["date"].unique()))
    return panel, calendar


@pytest.fixture(scope="module")
def seq_index(panel_and_calendar):
    panel, calendar = panel_and_calendar
    return build_sequence_index(panel, calendar=calendar)


@pytest.fixture(scope="module")
def fold_t300(panel_and_calendar):
    _, calendar = panel_and_calendar
    return fold_for_formation_session(
        T=300, calendar=calendar, train_window=config.TRAIN_WINDOW, horizon=config.FORECAST_HORIZON, fold_id=0
    )


# ---------------------------------------------------------------------------
# C. Exact boundary — the literal T=300 numeric proof requested.
# ---------------------------------------------------------------------------
def test_exact_boundary_t300(fold_t300, seq_index):
    fold = fold_t300
    candidate_end_range = [fold.train_window_first_session, fold.train_window_last_session]
    label_eligible_range = [fold.train_window_first_session, fold.label_eligible_last_session]

    assert candidate_end_range == [49, 300]          # 252 sessions: 300-49+1 == 252
    assert label_eligible_range == [49, 279]          # after +21 purge
    assert fold.train_window_last_session - fold.train_window_first_session + 1 == config.TRAIN_WINDOW
    assert fold.label_eligible_last_session == fold.formation_session - config.FORECAST_HORIZON

    endpoint_279_allowed = bool(279 + config.FORECAST_HORIZON <= fold.formation_session)
    endpoint_280_rejected = bool(280 + config.FORECAST_HORIZON > fold.formation_session)
    assert endpoint_279_allowed is True     # 279+21=300 == T
    assert endpoint_280_rejected is True    # 280+21=301 > T

    fs = build_lstm_fold_sequences(seq_index, fold)
    end_positions_in_train = set(fs.train[SEQ_END_POS_COL].unique())
    assert 279 in end_positions_in_train
    assert 280 not in end_positions_in_train


# ---------------------------------------------------------------------------
# A. Reading (A) — pre-window causal context allowed when it genuinely exists.
# ---------------------------------------------------------------------------
def test_sample_inside_window_may_use_causal_context_preceding_the_window(panel_and_calendar, seq_index, fold_t300):
    panel, _ = panel_and_calendar
    fold = fold_t300
    s = 100  # inside [49, 279]; its 60-session reach is [41, 100] — 41..48 precede window start (49)
    assert fold.train_window_first_session <= s <= fold.label_eligible_last_session
    causal_start = s - config.LSTM_SEQ + 1
    assert causal_start < fold.train_window_first_session  # 41 < 49: genuinely reaches before the window

    fs = build_lstm_fold_sequences(seq_index, fold)
    row = fs.train[(fs.train[SEQ_TICKER_COL] == "AAA") & (fs.train[SEQ_END_POS_COL] == s)]
    assert len(row) == 1
    assert int(row.iloc[0][SEQ_START_POS_COL]) == causal_start

    arr = extract_sequence_array(panel, row)
    assert arr.shape == (1, config.LSTM_SEQ, len(LSTM_FEATURE_COLS))
    # First timestep of the array is genuinely position 41 — real history, not fabricated.
    expected_first_row = [0 * 10000 + causal_start * 10 + j for j in range(len(LSTM_FEATURE_COLS))]
    np.testing.assert_array_equal(arr[0, 0], expected_first_row)


# ---------------------------------------------------------------------------
# B. No fabricated pre-history.
# ---------------------------------------------------------------------------
def test_no_fabricated_pre_history_within_nominal_candidate_window(seq_index, fold_t300):
    fold = fold_t300
    # Candidate window nominally starts at session 49, but LSTM_SEQ=60 means
    # no REAL sequence can end before session 59 (would need session -10..).
    # Sessions 49..58 are inside the nominal candidate window yet have zero
    # genuine 60-session history -> must be entirely absent, never wrapped
    # or padded.
    assert config.LSTM_SEQ - 1 > fold.train_window_first_session  # 59 > 49: this gap is real for T=300

    fs = build_lstm_fold_sequences(seq_index, fold)
    end_positions = fs.train[SEQ_END_POS_COL]
    assert not (end_positions < config.LSTM_SEQ - 1).any()
    assert (fs.train[SEQ_START_POS_COL] >= 0).all()  # no negative-index wraparound anywhere
    assert (seq_index[SEQ_START_POS_COL] >= 0).all()


def test_sequence_index_never_produces_negative_start_positions(panel_and_calendar):
    """A ticker whose history begins at session 0: no window can ever be
    built for an end position < LSTM_SEQ-1, however early a candidate
    window nominally starts."""
    panel, calendar = panel_and_calendar
    idx = build_sequence_index(panel, calendar=calendar)
    assert idx[SEQ_START_POS_COL].min() == 0
    assert idx[SEQ_END_POS_COL].min() == config.LSTM_SEQ - 1
    assert (idx[SEQ_START_POS_COL] >= 0).all()


# ---------------------------------------------------------------------------
# D. Sequence boundary for an eligible endpoint.
# ---------------------------------------------------------------------------
def test_training_row_sequence_is_exactly_60_and_bounded_by_endpoint(panel_and_calendar, seq_index, fold_t300):
    panel, _ = panel_and_calendar
    fs = build_lstm_fold_sequences(seq_index, fold_t300)
    row = fs.train.iloc[[0]]
    s = int(row.iloc[0][SEQ_END_POS_COL])
    assert int(row.iloc[0][SEQ_START_POS_COL]) == s - config.LSTM_SEQ + 1
    arr = extract_sequence_array(panel, row)
    assert arr.shape == (1, config.LSTM_SEQ, len(LSTM_FEATURE_COLS))


# ---------------------------------------------------------------------------
# E. Outer OOS.
# ---------------------------------------------------------------------------
def test_outer_oos_uses_exactly_T_minus_59_to_T_and_never_leaks_into_training(
    panel_and_calendar, seq_index, fold_t300
):
    panel, _ = panel_and_calendar
    fold = fold_t300
    fs = build_lstm_fold_sequences(seq_index, fold)

    assert set(fs.outer_oos[GLOBAL_SESSION_COL].unique()) == {fold.eval_session}
    assert fold.eval_session == 300
    oos_row = fs.outer_oos[fs.outer_oos[SEQ_TICKER_COL] == "AAA"]
    assert int(oos_row.iloc[0][SEQ_START_POS_COL]) == 300 - config.LSTM_SEQ + 1  # == 241
    assert int(oos_row.iloc[0][SEQ_END_POS_COL]) == 300

    # OOS session (300) never appears in any training/internal-validation bucket.
    for bucket in (fs.train, fs.inner_train, fs.inner_val):
        assert fold.eval_session not in set(bucket[GLOBAL_SESSION_COL].unique())

    # No future (session 301+) feature enters the OOS array.
    before = extract_sequence_array(panel, oos_row)
    mutated = panel.copy()
    mutated.loc[mutated["date"] > oos_row.iloc[0]["seq_end_date"], LSTM_FEATURE_COLS] = -999999.0
    after = extract_sequence_array(mutated, oos_row)
    np.testing.assert_array_equal(before, after)


# ---------------------------------------------------------------------------
# F. Pooled training across tickers.
# ---------------------------------------------------------------------------
def test_pooled_training_combines_tickers_and_retains_identity(seq_index, fold_t300):
    fs = build_lstm_fold_sequences(seq_index, fold_t300)
    assert set(fs.train[SEQ_TICKER_COL].unique()) == set(TICKERS)
    assert fs.train[[SEQ_TICKER_COL, GLOBAL_SESSION_COL]].notna().all().all()


# ---------------------------------------------------------------------------
# G. Internal temporal validation.
# ---------------------------------------------------------------------------
def test_internal_validation_ordering_and_outer_oos_absence(seq_index, fold_t300):
    fold = fold_t300
    fs = build_lstm_fold_sequences(seq_index, fold, val_sessions=21)

    assert len(fs.inner_train) > 0
    assert len(fs.inner_val) > 0
    assert fs.inner_train[GLOBAL_SESSION_COL].max() < fs.inner_val[GLOBAL_SESSION_COL].min()
    assert fs.inner_val[GLOBAL_SESSION_COL].max() == fold.label_eligible_last_session
    assert fs.inner_val[GLOBAL_SESSION_COL].max() < fold.eval_session  # outer-OOS-blind

    # Every inner_train/inner_val row still satisfies the SAME label-
    # availability rule as the outer training set (inherited, not
    # separately invented).
    for bucket in (fs.inner_train, fs.inner_val):
        assert (bucket[GLOBAL_SESSION_COL] + config.FORECAST_HORIZON <= fold.formation_session).all()


# ---------------------------------------------------------------------------
# H. Future perturbation at the fold level.
# ---------------------------------------------------------------------------
def test_earlier_fold_unaffected_by_perturbing_much_later_sessions(panel_and_calendar):
    panel, calendar = panel_and_calendar
    idx_before = build_sequence_index(panel, calendar=calendar)
    fold_early = fold_for_formation_session(
        T=150, calendar=calendar, train_window=config.TRAIN_WINDOW, horizon=config.FORECAST_HORIZON, fold_id=0
    )
    fs_before = build_lstm_fold_sequences(idx_before, fold_early)

    mutated = panel.copy()
    cutoff_date = calendar[200]
    mutated.loc[mutated["date"] > cutoff_date, LSTM_FEATURE_COLS] = -12345.0
    mutated.loc[mutated["date"] > cutoff_date, TARGET_COL] = -12345.0

    idx_after = build_sequence_index(mutated, calendar=calendar)
    fs_after = build_lstm_fold_sequences(idx_after, fold_early)

    key_cols = [SEQ_TICKER_COL, GLOBAL_SESSION_COL]
    for name in ("train", "inner_train", "inner_val", "outer_oos"):
        before_keys = getattr(fs_before, name)[key_cols].reset_index(drop=True)
        after_keys = getattr(fs_after, name)[key_cols].reset_index(drop=True)
        pd.testing.assert_frame_equal(before_keys, after_keys)

    arr_before = extract_sequence_array(panel, fs_before.train)
    arr_after = extract_sequence_array(mutated, fs_after.train)
    np.testing.assert_array_equal(arr_before, arr_after)
