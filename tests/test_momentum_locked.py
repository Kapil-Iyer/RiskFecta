"""
RiskFecta Phase 3 — momentum decision-gate LOCKING tests.

The momentum session-count decision gate (pipeline/features.py module
docstring, Decision 1) is now explicitly locked by the user:
    config.MOMENTUM_3M_SESSIONS = 63
    config.MOMENTUM_6M_SESSIONS = 126

This file proves the seven properties the delta prompt requires:
  1. production/default Phase 3 wiring uses exactly 63 and 126
  2. momentum_3m[t] = close[t] / close[t-63] - 1
  3. momentum_6m[t] = close[t] / close[t-126] - 1
  4. no off-by-one
  5. warm-up rows remain unavailable
  6. future price perturbations do not change already-computable historical momentum
  7. ticker isolation remains intact

No calendar-month arithmetic is used anywhere — purely valid-session offsets.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import config
from pipeline.features import build_feature_frame, compute_technical_features


def _prices(ticker: str, closes, start="2021-01-04"):
    dates = pd.bdate_range(start, periods=len(closes))
    return pd.DataFrame(
        {
            "ticker": ticker,
            "date": dates,
            "close": closes,
            "total_return_idx": closes,
        }
    )


def _macro(dates):
    return pd.DataFrame({"date": dates, "vix": [20.0] * len(dates), "yield_10y": [1.5] * len(dates)})


N = 200  # long enough to clear both the 63- and 126-session warm-ups with margin


def test_config_constants_are_exactly_63_and_126():
    assert config.MOMENTUM_3M_SESSIONS == 63
    assert config.MOMENTUM_6M_SESSIONS == 126


def test_default_wiring_uses_exactly_63_and_126_no_override():
    """1. production/default Phase 3 wiring uses exactly 63 and 126."""
    closes = [100 + i + 0.3 * np.sin(i / 5) for i in range(N)]
    prices = _prices("AAPL", closes)

    out = compute_technical_features(prices)  # no momentum_*_sessions passed at all

    # Same as the config-driven default, computed independently with the
    # generic function using an explicit 63/126 -- must match exactly.
    out_explicit = compute_technical_features(
        prices, momentum_3m_sessions=63, momentum_6m_sessions=126
    )
    np.testing.assert_array_equal(out["momentum_3m"].values, out_explicit["momentum_3m"].values)
    np.testing.assert_array_equal(out["momentum_6m"].values, out_explicit["momentum_6m"].values)

    # And must NOT match some other arbitrary lookback (guards against a
    # silently-wrong constant sneaking into the default wiring).
    out_wrong = compute_technical_features(prices, momentum_3m_sessions=21, momentum_6m_sessions=42)
    assert not np.allclose(
        out["momentum_3m"].dropna().values, out_wrong["momentum_3m"].dropna().values[: out["momentum_3m"].dropna().shape[0]]
    ) or out["momentum_3m"].dropna().shape[0] != out_wrong["momentum_3m"].dropna().shape[0]


def test_momentum_3m_formula_close_t_over_close_t_minus_63():
    """2. momentum_3m[t] = close[t] / close[t-63] - 1."""
    closes = [100 + i + 0.3 * np.sin(i / 5) for i in range(N)]
    prices = _prices("AAPL", closes)
    out = compute_technical_features(prices)

    for t in range(63, N):
        expected = closes[t] / closes[t - 63] - 1
        assert out.loc[t, "momentum_3m"] == pytest.approx(expected, rel=1e-12)


def test_momentum_6m_formula_close_t_over_close_t_minus_126():
    """3. momentum_6m[t] = close[t] / close[t-126] - 1."""
    closes = [100 + i + 0.3 * np.sin(i / 5) for i in range(N)]
    prices = _prices("AAPL", closes)
    out = compute_technical_features(prices)

    for t in range(126, N):
        expected = closes[t] / closes[t - 126] - 1
        assert out.loc[t, "momentum_6m"] == pytest.approx(expected, rel=1e-12)


def test_no_off_by_one_at_the_exact_boundary():
    """4. no off-by-one: the first available momentum_3m is at index exactly
    63 (using close[0]), not 62 or 64; same for momentum_6m at index 126.
    """
    closes = list(range(100, 100 + N))
    prices = _prices("AAPL", closes)
    out = compute_technical_features(prices)

    assert pd.isna(out.loc[62, "momentum_3m"])
    assert out.loc[63, "momentum_3m"] == pytest.approx(closes[63] / closes[0] - 1)

    assert pd.isna(out.loc[125, "momentum_6m"])
    assert out.loc[126, "momentum_6m"] == pytest.approx(closes[126] / closes[0] - 1)


def test_warmup_rows_remain_unavailable():
    """5. warm-up rows remain unavailable (NaN, never 0 or back-filled)."""
    closes = [100 + i for i in range(N)]
    prices = _prices("AAPL", closes)
    out = compute_technical_features(prices)

    assert out["momentum_3m"].iloc[:63].isna().all()
    assert (out["momentum_3m"].iloc[:63] == 0).sum() == 0
    assert out["momentum_6m"].iloc[:126].isna().all()
    assert (out["momentum_6m"].iloc[:126] == 0).sum() == 0


def test_future_perturbation_does_not_change_historical_momentum():
    """6. future price perturbations do not change already-computable
    historical momentum_3m/momentum_6m."""
    closes = [100 + i + 0.3 * np.sin(i / 5) for i in range(N)]
    prices = _prices("AAPL", closes)
    before = compute_technical_features(prices)

    cutoff = 140
    perturbed_closes = list(closes)
    for i in range(cutoff + 1, N):
        perturbed_closes[i] += 50_000
    perturbed_prices = _prices("AAPL", perturbed_closes)
    after = compute_technical_features(perturbed_prices)

    np.testing.assert_allclose(
        before["momentum_3m"].iloc[: cutoff + 1].values.astype(float),
        after["momentum_3m"].iloc[: cutoff + 1].values.astype(float),
        equal_nan=True,
    )
    np.testing.assert_allclose(
        before["momentum_6m"].iloc[: cutoff + 1].values.astype(float),
        after["momentum_6m"].iloc[: cutoff + 1].values.astype(float),
        equal_nan=True,
    )


def test_ticker_isolation_intact_under_default_wiring():
    """7. ticker isolation remains intact under the locked default wiring."""
    aapl_closes = [100 + i + 0.3 * np.sin(i / 5) for i in range(N)]
    msft_closes = [500 - 0.5 * i + 0.2 * np.cos(i / 7) for i in range(N)]
    aapl = _prices("AAPL", aapl_closes)
    msft = _prices("MSFT", msft_closes)
    combined = pd.concat([aapl, msft]).sample(frac=1.0, random_state=5).reset_index(drop=True)

    out_combined = compute_technical_features(combined)
    out_aapl_alone = compute_technical_features(aapl)
    out_msft_alone = compute_technical_features(msft)

    for col in ("momentum_3m", "momentum_6m"):
        a = out_combined[out_combined["ticker"] == "AAPL"][col].reset_index(drop=True)
        b = out_aapl_alone[col].reset_index(drop=True)
        np.testing.assert_allclose(a.values.astype(float), b.values.astype(float), equal_nan=True)

        m = out_combined[out_combined["ticker"] == "MSFT"][col].reset_index(drop=True)
        n = out_msft_alone[col].reset_index(drop=True)
        np.testing.assert_allclose(m.values.astype(float), n.values.astype(float), equal_nan=True)


def test_build_feature_frame_default_wiring_end_to_end():
    """The full production orchestration path (build_feature_frame, no
    momentum overrides) must also use the locked 63/126 constants.
    """
    closes = [100 + i + 0.3 * np.sin(i / 5) for i in range(N)]
    prices = _prices("AAPL", closes)
    macro = _macro(prices["date"])

    out = build_feature_frame(prices, macro)  # no momentum args at all

    assert out["momentum_3m"].iloc[63:].notna().all()
    assert out["momentum_6m"].iloc[126:].notna().all()
    for t in (63, 100, 199):
        assert out.loc[t, "momentum_3m"] == pytest.approx(closes[t] / closes[t - 63] - 1)
    for t in (126, 150, 199):
        assert out.loc[t, "momentum_6m"] == pytest.approx(closes[t] / closes[t - 126] - 1)
