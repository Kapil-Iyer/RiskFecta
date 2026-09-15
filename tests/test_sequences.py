"""
RiskFecta Phase 5A — sequence-builder tests (pipeline/sequences.py).

Synthetic/adversarial only (task brief §20, §21) — no real Bloomberg data
anywhere in this file. Every LSTM feature value is a deterministic function
of (ticker index, ticker-local position, column index) so extraction
correctness can be checked by exact reconstruction, not just shape.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import config
from pipeline.folds import GLOBAL_SESSION_COL, internal_validation_split, rows_in_session_range
from pipeline.sequences import (
    LSTM_FEATURE_COLS,
    SEQ_END_DATE_COL,
    SEQ_END_POS_COL,
    SEQ_START_DATE_COL,
    SEQ_START_POS_COL,
    SEQ_TARGET_COL,
    SEQ_TICKER_COL,
    SEQUENCE_LENGTH,
    build_sequence_index,
    extract_sequence_array,
    label_eligible_mask,
)
from pipeline.targets import TARGET_COL


def _make_panel(tickers, n_sessions, warmup=0, start="2021-01-04", target_horizon=21):
    """Deterministic synthetic features+target panel. Feature value at
    (ticker t_i, position p, column j) = t_i*10000 + p*10 + j — unique and
    exactly reconstructable, so extraction bugs (off-by-one, wrong ticker,
    wrong direction) are directly detectable, not just shape-checked.
    `date` uses business days (skips weekends), so calendar span always
    exceeds session count by construction (task brief §20-D).

    `warmup` NaNs only the TECHNICAL indicator columns
    (`config.LSTM_FEATURE_COLS` — rsi_14/macd/.../momentum_6m), never the
    raw OHLCV price columns (`config.LSTM_PRICE_COLS`) — matching real
    Phase 3 semantics exactly: a technical indicator's warm-up NaN occurs
    on an otherwise perfectly VALID trading session (the session-validity
    gate is `close`, an OHLCV price column, per `pipeline.sessions`/ML_SPEC
    §3 — it is never NaN merely because RSI/MACD haven't warmed up yet).
    """
    dates = pd.bdate_range(start=start, periods=n_sessions)
    frames = []
    for t_i, ticker in enumerate(tickers):
        pos = np.arange(n_sessions)
        df = pd.DataFrame({"ticker": ticker, "date": dates})
        for j, col in enumerate(LSTM_FEATURE_COLS):
            df[col] = t_i * 10000 + pos * 10 + j
        if warmup > 0:
            df.loc[: warmup - 1, config.LSTM_FEATURE_COLS] = np.nan
        tri = 100.0 + pos * 0.05
        target = pd.Series(tri).shift(-target_horizon) / pd.Series(tri) - 1
        df[TARGET_COL] = target.to_numpy()
        frames.append(df)
    panel = pd.concat(frames, ignore_index=True)
    return panel.sort_values(["ticker", "date"], kind="mergesort").reset_index(drop=True)


def _expected_value(ticker_idx: int, pos: int, col_idx: int) -> float:
    return float(ticker_idx * 10000 + pos * 10 + col_idx)


# ---------------------------------------------------------------------------
# I. Feature surface (locked)
# ---------------------------------------------------------------------------
def test_lstm_feature_surface_matches_config():
    assert LSTM_FEATURE_COLS == list(config.LSTM_ALL_FEATURES)
    assert LSTM_FEATURE_COLS == list(config.LSTM_PRICE_COLS) + list(config.LSTM_FEATURE_COLS)
    assert len(LSTM_FEATURE_COLS) == 13


def test_forbidden_static_and_macro_columns_rejected():
    panel = _make_panel(["AAA"], 70)
    with pytest.raises(ValueError):
        build_sequence_index(panel, feature_cols=LSTM_FEATURE_COLS + ["beta"])
    with pytest.raises(ValueError):
        build_sequence_index(panel, feature_cols=LSTM_FEATURE_COLS + ["vix"])


# ---------------------------------------------------------------------------
# A. Exact sequence length
# ---------------------------------------------------------------------------
def test_every_sequence_is_exactly_seq_len_long():
    panel = _make_panel(["AAA"], 100)
    idx = build_sequence_index(panel)
    assert len(idx) > 0
    lengths = idx[SEQ_END_POS_COL] - idx[SEQ_START_POS_COL] + 1
    assert (lengths == SEQUENCE_LENGTH).all()
    assert SEQUENCE_LENGTH == config.LSTM_SEQ == 60


# ---------------------------------------------------------------------------
# B. Same-ticker isolation
# ---------------------------------------------------------------------------
def test_sequences_never_cross_ticker_boundaries():
    tickers = ["AAA", "BBB"]
    panel = _make_panel(tickers, 90)
    idx = build_sequence_index(panel)
    arr = extract_sequence_array(panel, idx)

    for row_i, rec in enumerate(idx.itertuples(index=False)):
        t_i = tickers.index(getattr(rec, SEQ_TICKER_COL))
        start_pos = getattr(rec, SEQ_START_POS_COL)
        expected = np.array(
            [[_expected_value(t_i, start_pos + k, j) for j in range(len(LSTM_FEATURE_COLS))] for k in range(SEQUENCE_LENGTH)]
        )
        np.testing.assert_array_equal(arr[row_i], expected)


# ---------------------------------------------------------------------------
# C. Chronological order
# ---------------------------------------------------------------------------
def test_sequence_dates_are_strictly_chronological():
    panel = _make_panel(["AAA"], 80)
    idx = build_sequence_index(panel)
    dates = pd.to_datetime(panel.loc[panel[SEQ_TICKER_COL] == "AAA", "date"]).sort_values().reset_index(drop=True)
    for rec in idx.itertuples(index=False):
        start_pos, end_pos = getattr(rec, SEQ_START_POS_COL), getattr(rec, SEQ_END_POS_COL)
        window_dates = dates.iloc[start_pos : end_pos + 1]
        assert window_dates.is_monotonic_increasing
        assert window_dates.iloc[-1] == getattr(rec, SEQ_END_DATE_COL)
        assert window_dates.iloc[0] == getattr(rec, SEQ_START_DATE_COL)


# ---------------------------------------------------------------------------
# D. Session vs. calendar
# ---------------------------------------------------------------------------
def test_calendar_gaps_do_not_change_60_session_semantics():
    panel = _make_panel(["AAA"], 80)  # business days -> weekends already excluded
    idx = build_sequence_index(panel)
    row = idx.iloc[0]
    span_days = (row[SEQ_END_DATE_COL] - row[SEQ_START_DATE_COL]).days
    # 60 valid (business-day) sessions span more than 59 calendar days once
    # weekends are counted -> proves the 60-count is session-based, not a
    # fixed calendar-day window.
    assert span_days > SEQUENCE_LENGTH - 1


# ---------------------------------------------------------------------------
# E. No future features
# ---------------------------------------------------------------------------
def test_perturbing_future_rows_does_not_change_earlier_sequences():
    panel = _make_panel(["AAA"], 90)
    idx = build_sequence_index(panel)
    early_row = idx.iloc[0]
    before = extract_sequence_array(panel, idx.iloc[[0]])

    mutated = panel.copy()
    future_mask = mutated["date"] > early_row[SEQ_END_DATE_COL]
    assert future_mask.any()
    mutated.loc[future_mask, LSTM_FEATURE_COLS] = 999999.0

    idx_after = build_sequence_index(mutated)
    after = extract_sequence_array(mutated, idx_after.iloc[[0]])
    np.testing.assert_array_equal(before, after)


# ---------------------------------------------------------------------------
# F. Target alignment
# ---------------------------------------------------------------------------
def test_target_alignment_matches_horizon_shift():
    panel = _make_panel(["AAA"], 90, target_horizon=config.FORECAST_HORIZON)
    idx = build_sequence_index(panel)
    merged = idx.merge(
        panel[["ticker", "date", TARGET_COL]].rename(columns={"date": SEQ_END_DATE_COL}),
        on=[SEQ_TICKER_COL, SEQ_END_DATE_COL],
        how="left",
    )
    # target carried in the sequence index must equal the panel's own
    # target_21d at the sequence's END date (never any other row's).
    # SEQ_TARGET_COL == TARGET_COL == "target_21d", so the merge suffixes
    # both copies as "_x" (from idx) / "_y" (freshly joined from panel).
    left_col, right_col = f"{SEQ_TARGET_COL}_x", f"{TARGET_COL}_y"
    both_notna = merged[left_col].notna() & merged[right_col].notna()
    assert both_notna.any()
    np.testing.assert_allclose(merged.loc[both_notna, left_col], merged.loc[both_notna, right_col])


# ---------------------------------------------------------------------------
# G. Label availability (endpoint boundary)
# ---------------------------------------------------------------------------
def test_label_eligible_mask_exact_boundary_and_one_after():
    panel = _make_panel(["AAA"], 90)
    calendar = pd.DatetimeIndex(sorted(panel["date"].unique()))
    idx = build_sequence_index(panel, calendar=calendar)
    assert idx[GLOBAL_SESSION_COL].notna().all()

    sample = idx.iloc[10]
    gs = int(sample[GLOBAL_SESSION_COL])
    horizon = config.FORECAST_HORIZON

    # Endpoint exactly on T -> allowed.
    mask_exact = label_eligible_mask(idx, formation_session=gs + horizon, horizon=horizon)
    assert bool(mask_exact.iloc[10]) is True

    # One session after T -> rejected.
    mask_one_after = label_eligible_mask(idx, formation_session=gs + horizon - 1, horizon=horizon)
    assert bool(mask_one_after.iloc[10]) is False


def test_label_eligible_mask_requires_global_session():
    panel = _make_panel(["AAA"], 90)
    idx = build_sequence_index(panel)  # no calendar -> global_session all NaN
    with pytest.raises(ValueError):
        label_eligible_mask(idx, formation_session=100)


# ---------------------------------------------------------------------------
# J. Missing-data eligibility (no fill)
# ---------------------------------------------------------------------------
def test_warmup_nulls_exclude_only_windows_that_need_them():
    n_sessions, warmup = 200, 70
    panel = _make_panel(["AAA"], n_sessions, warmup=warmup)
    idx = build_sequence_index(panel)
    first_eligible_end = warmup + SEQUENCE_LENGTH - 1
    expected_count = n_sessions - first_eligible_end
    assert len(idx) == expected_count
    assert idx[SEQ_END_POS_COL].min() == first_eligible_end


def test_isolated_missing_row_excludes_only_overlapping_windows():
    panel = _make_panel(["AAA"], 100)
    gap_pos = 50
    panel.loc[gap_pos, LSTM_FEATURE_COLS[0]] = np.nan  # single missing feature, single row
    idx = build_sequence_index(panel)
    overlapping = idx[(idx[SEQ_START_POS_COL] <= gap_pos) & (idx[SEQ_END_POS_COL] >= gap_pos)]
    assert len(overlapping) == 0
    not_overlapping = idx[(idx[SEQ_END_POS_COL] < gap_pos) | (idx[SEQ_START_POS_COL] > gap_pos)]
    assert len(not_overlapping) == len(idx)
    # No fill occurred anywhere: the panel's own NaN is untouched.
    assert pd.isna(panel.loc[gap_pos, LSTM_FEATURE_COLS[0]])


# ---------------------------------------------------------------------------
# K. Pooled training combination / L. Evaluation identity
# ---------------------------------------------------------------------------
def test_pooled_index_combines_all_tickers_and_retains_identity():
    tickers = ["AAA", "BBB", "CCC"]
    panel = _make_panel(tickers, 90)
    idx = build_sequence_index(panel)
    assert set(idx[SEQ_TICKER_COL].unique()) == set(tickers)

    per_ticker_counts = {t: len(build_sequence_index(panel[panel["ticker"] == t])) for t in tickers}
    assert len(idx) == sum(per_ticker_counts.values())

    arr = extract_sequence_array(panel, idx)
    assert arr.shape == (len(idx), SEQUENCE_LENGTH, len(LSTM_FEATURE_COLS))
    # identity preserved: every row of idx still carries its own ticker/date.
    assert idx[[SEQ_TICKER_COL, SEQ_END_DATE_COL]].notna().all().all()


def test_empty_panel_returns_empty_index():
    panel = _make_panel(["AAA"], 10)  # fewer than seq_len -> no eligible windows
    idx = build_sequence_index(panel)
    assert len(idx) == 0
    arr = extract_sequence_array(panel, idx)
    assert arr.shape[0] == 0


# ---------------------------------------------------------------------------
# N. Internal temporal validation — reuses pipeline.folds infra directly
# (no new purge logic invented for LSTM; see pipeline.sequences module
# docstring re: the still-open TRAIN_WINDOW/LSTM_SEQ fold-wiring gate).
# ---------------------------------------------------------------------------
def test_internal_validation_split_reuses_fold_infrastructure():
    from pipeline.folds import fold_for_formation_session

    tickers = ["AAA", "BBB"]
    panel = _make_panel(tickers, 300)
    calendar = pd.DatetimeIndex(sorted(panel["date"].unique()))
    idx = build_sequence_index(panel, calendar=calendar)

    # A realistic Fold (T=290) built via the SAME pipeline.folds constructor
    # Phase 4A already uses — no LSTM-specific purge logic invented. Chosen
    # so its label_eligible_last_session (269) is comfortably inside the
    # region where target_21d is realized (valid through position 278 given
    # target_horizon=21 default and n_sessions=300).
    fold = fold_for_formation_session(
        T=290, calendar=calendar, train_window=config.TRAIN_WINDOW, horizon=config.FORECAST_HORIZON, fold_id=0
    )
    inner_train_range, inner_val_range = internal_validation_split(fold, val_sessions=21)

    inner_train = rows_in_session_range(idx, inner_train_range, feature_cols=[], target_col=SEQ_TARGET_COL)
    inner_val = rows_in_session_range(idx, inner_val_range, feature_cols=[], target_col=SEQ_TARGET_COL)

    assert len(inner_train) > 0
    assert len(inner_val) > 0
    assert inner_train[GLOBAL_SESSION_COL].max() < inner_val[GLOBAL_SESSION_COL].min()
    assert inner_val[GLOBAL_SESSION_COL].max() == fold.label_eligible_last_session
    assert inner_val[GLOBAL_SESSION_COL].max() < fold.eval_session  # outer-OOS-blind
