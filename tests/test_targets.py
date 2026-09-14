"""
RiskFecta Phase 3 — target alignment tests (pipeline/targets.py).

Covers task brief §18-A (target alignment) and §18-L/§18-M (future-
perturbation causality, target/feature separation).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pipeline.targets import TARGET_COL, compute_targets


def _synthetic_ticker(ticker: str, n: int, start_tri: float = 100.0, step: float = 1.0) -> pd.DataFrame:
    """n valid sessions, TRI = start_tri + step*i (a simple, hand-verifiable
    sequence), dates strictly ascending business days.
    """
    dates = pd.bdate_range("2021-01-04", periods=n)
    tri = [start_tri + step * i for i in range(n)]
    return pd.DataFrame(
        {
            "ticker": ticker,
            "date": dates,
            "close": [100.0 + i for i in range(n)],  # close just needs to be non-null (session gate)
            "total_return_idx": tri,
        }
    )


def test_target_formula_exact_known_answer():
    # TRI increases by 1 each session starting at 100: TRI[t]=100+t.
    # y(t) = TRI[t+21]/TRI[t] - 1 = (100+t+21)/(100+t) - 1, hand-computable.
    df = _synthetic_ticker("AAPL", n=30)
    out = compute_targets(df, horizon=21)

    for t in range(30 - 21):
        expected = (100 + t + 21) / (100 + t) - 1
        got = out.loc[t, TARGET_COL]
        assert got == pytest.approx(expected, rel=1e-12), f"t={t}: expected {expected}, got {got}"


def test_exact_21_session_offset_no_off_by_one():
    df = _synthetic_ticker("AAPL", n=25)
    out = compute_targets(df, horizon=21)

    # t=0 target must reference row index 21 (the 22nd row), not 20 or 22.
    tri = df["total_return_idx"].values
    assert out.loc[0, TARGET_COL] == pytest.approx(tri[21] / tri[0] - 1)
    with pytest.raises(AssertionError):
        assert out.loc[0, TARGET_COL] == pytest.approx(tri[20] / tri[0] - 1)
    with pytest.raises(AssertionError):
        assert out.loc[0, TARGET_COL] == pytest.approx(tri[22] / tri[0] - 1)


def test_final_21_sessions_have_no_target():
    n = 30
    df = _synthetic_ticker("AAPL", n=n)
    out = compute_targets(df, horizon=21)

    # Rows n-21 .. n-1 (the final 21 valid sessions) must be NaN.
    tail = out.iloc[n - 21 :]
    assert tail[TARGET_COL].isna().all()
    # Everything before that must be defined (TRI has no missing values here).
    head = out.iloc[: n - 21]
    assert head[TARGET_COL].notna().all()


def test_return_direction_is_correct_not_inverted():
    # Rising TRI must produce a positive target; falling TRI a negative one.
    rising = _synthetic_ticker("AAPL", n=25, start_tri=100.0, step=1.0)
    falling = _synthetic_ticker("MSFT", n=25, start_tri=100.0, step=-1.0)
    df = pd.concat([rising, falling], ignore_index=True)
    out = compute_targets(df, horizon=21)

    assert (out.loc[out["ticker"] == "AAPL", TARGET_COL].dropna() > 0).all()
    assert (out.loc[out["ticker"] == "MSFT", TARGET_COL].dropna() < 0).all()


def test_weekend_tri_populated_does_not_shift_target():
    """A weekend/holiday calendar row with close=NaN but TOTAL_RETURN_INDEX
    still populated must be excluded from the valid-session sequence
    entirely — it must never occupy a 't' or 't+21' slot, and must never
    shift the target computed from the surrounding genuine sessions.
    """
    valid = _synthetic_ticker("AAPL", n=25)  # 25 valid sessions, TRI = 100..124
    # Insert a weekend placeholder row between session index 4 and 5:
    # close = NaN (invalid), TRI populated (Bloomberg quirk) at a date between
    # the two sessions' actual dates.
    weekend_date = valid.loc[4, "date"] + pd.Timedelta(hours=12)
    weekend_row = pd.DataFrame(
        [{"ticker": "AAPL", "date": weekend_date, "close": np.nan, "total_return_idx": 9999.0}]
    )
    with_weekend = pd.concat([valid, weekend_row], ignore_index=True).sort_values("date").reset_index(drop=True)

    out_valid = compute_targets(valid, horizon=21)
    out_with_weekend = compute_targets(with_weekend, horizon=21)

    # Identical target sequence regardless of the injected weekend row.
    np.testing.assert_allclose(
        out_valid[TARGET_COL].values.astype(float),
        out_with_weekend[TARGET_COL].values.astype(float),
        equal_nan=True,
    )
    # And the weekend row itself never appears in the output.
    assert weekend_date not in out_with_weekend["date"].values


def test_ticker_isolation_no_cross_ticker_target_leakage():
    aapl = _synthetic_ticker("AAPL", n=25, start_tri=100.0, step=1.0)
    msft = _synthetic_ticker("MSFT", n=25, start_tri=500.0, step=2.0)
    # Interleave rows to catch an accidental ungrouped shift.
    combined = pd.concat([aapl, msft]).sort_values("date").reset_index(drop=True)

    out = compute_targets(combined, horizon=21)
    aapl_only = compute_targets(aapl, horizon=21)
    msft_only = compute_targets(msft, horizon=21)

    out_aapl = out[out["ticker"] == "AAPL"].reset_index(drop=True)
    out_msft = out[out["ticker"] == "MSFT"].reset_index(drop=True)

    np.testing.assert_allclose(
        out_aapl[TARGET_COL].values.astype(float), aapl_only[TARGET_COL].values.astype(float), equal_nan=True
    )
    np.testing.assert_allclose(
        out_msft[TARGET_COL].values.astype(float), msft_only[TARGET_COL].values.astype(float), equal_nan=True
    )


def test_future_perturbation_causality_for_targets_is_expected_to_change_reachable_targets():
    """Unlike features, a target IS allowed (and expected) to change when a
    future TRI value used in its own t+21 lookup changes — this is the
    documented target/feature-separation distinction (task brief §18-M),
    tested here so it's never confused with a leakage bug.
    """
    df = _synthetic_ticker("AAPL", n=30)
    before = compute_targets(df, horizon=21)

    perturbed = df.copy()
    perturbed.loc[25, "total_return_idx"] = perturbed.loc[25, "total_return_idx"] * 2  # far-future TRI
    after = compute_targets(perturbed, horizon=21)

    # t=4's target horizon reaches row 4+21=25 -> its target legitimately changes.
    assert before.loc[4, TARGET_COL] != after.loc[4, TARGET_COL]
    # t=3's target horizon reaches row 24 (untouched) -> must be unchanged.
    assert before.loc[3, TARGET_COL] == pytest.approx(after.loc[3, TARGET_COL])


def test_determinism_shuffled_row_order_same_sorted_result():
    df = _synthetic_ticker("AAPL", n=25)
    shuffled = df.sample(frac=1.0, random_state=7).reset_index(drop=True)

    out_sorted = compute_targets(df, horizon=21)
    out_shuffled = compute_targets(shuffled, horizon=21)

    pd.testing.assert_frame_equal(
        out_sorted.reset_index(drop=True), out_shuffled.reset_index(drop=True), check_exact=False
    )
