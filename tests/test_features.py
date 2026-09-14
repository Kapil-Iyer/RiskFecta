"""
RiskFecta Phase 3 — feature computation tests (pipeline/features.py).

Each technical indicator is checked against an independent, non-vectorized
reference loop (never just re-invoking the function under test), for
warm-up behavior, and for the future-perturbation causality property
(changing prices/macro strictly after a cutoff must never change any
feature value at or before that cutoff). Section J covers static-snapshot
leakage; the comprehensive combined causality test is
test_future_perturbation_causality_all_features (task brief §18-L, the
single most important Phase 3 test).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pipeline.features import (
    align_macro_to_sessions,
    build_feature_frame,
    compute_bollinger,
    compute_macd,
    compute_momentum,
    compute_rsi,
    compute_spx_return,
    compute_technical_features,
    compute_volatility,
    STATIC_LEAKAGE_COLS,
)
from pipeline.sessions import valid_sessions


def _prices(ticker: str, closes, start="2021-01-04"):
    dates = pd.bdate_range(start, periods=len(closes))
    return pd.DataFrame(
        {
            "ticker": ticker,
            "date": dates,
            "close": closes,
            "total_return_idx": closes,  # irrelevant to feature computation; harmless placeholder
        }
    )


# ---------------------------------------------------------------------------
# C. RSI
# ---------------------------------------------------------------------------
def _reference_rsi(closes, period=14):
    """Independent, explicit step-by-step recursion (Wilder EWM, no SMA
    seed) — deliberately NOT the vectorized implementation under test.
    """
    n = len(closes)
    out = [np.nan] * n
    avg_gain = 0.0
    avg_loss = 0.0
    alpha = 1.0 / period
    prev_close = closes[0]
    for t in range(1, n):
        delta = closes[t] - prev_close
        gain = max(delta, 0.0)
        loss = max(-delta, 0.0)
        avg_gain = (1 - alpha) * avg_gain + alpha * gain
        avg_loss = (1 - alpha) * avg_loss + alpha * loss
        if t >= period:
            if avg_loss == 0:
                out[t] = 100.0
            else:
                rs = avg_gain / avg_loss
                out[t] = 100 - 100 / (1 + rs)
        prev_close = closes[t]
    return out


def test_rsi_matches_independent_reference_and_warmup():
    closes = [100 + 0.7 * i - (i % 3) for i in range(40)]
    rsi = compute_rsi(pd.Series(closes), period=14)
    ref = _reference_rsi(closes, period=14)

    np.testing.assert_allclose(rsi.values, np.array(ref), equal_nan=True, atol=1e-9)
    assert rsi.iloc[:14].isna().all()
    assert rsi.iloc[14:].notna().all()


def test_rsi_no_future_dependency():
    closes = [100 + 0.7 * i - (i % 3) for i in range(40)]
    before = compute_rsi(pd.Series(closes), period=14)

    perturbed = list(closes)
    perturbed[35] += 500  # far-future perturbation
    after = compute_rsi(pd.Series(perturbed), period=14)

    np.testing.assert_allclose(before.iloc[:35].values, after.iloc[:35].values, equal_nan=True)


# ---------------------------------------------------------------------------
# D. MACD
# ---------------------------------------------------------------------------
def _reference_macd(closes, fast=12, slow=26, signal=9):
    n = len(closes)
    ema_fast = [np.nan] * n
    ema_slow = [np.nan] * n
    macd = [np.nan] * n
    alpha_f, alpha_s = 2 / (fast + 1), 2 / (slow + 1)
    ema_fast[0] = closes[0]
    ema_slow[0] = closes[0]
    for t in range(1, n):
        ema_fast[t] = (1 - alpha_f) * ema_fast[t - 1] + alpha_f * closes[t]
        ema_slow[t] = (1 - alpha_s) * ema_slow[t - 1] + alpha_s * closes[t]
    for t in range(n):
        macd[t] = ema_fast[t] - ema_slow[t] if t >= slow - 1 else np.nan

    signal_line = [np.nan] * n
    alpha_sig = 2 / (signal + 1)
    warm = slow - 1
    signal_line[warm] = macd[warm]
    for t in range(warm + 1, n):
        signal_line[t] = (1 - alpha_sig) * signal_line[t - 1] + alpha_sig * macd[t]
    warm2 = slow - 1 + signal - 1
    for t in range(min(warm2, n)):
        signal_line[t] = np.nan
    return macd, signal_line


def test_macd_matches_independent_reference_and_warmup():
    closes = [100 + 0.5 * i + 3 * np.sin(i / 4) for i in range(60)]
    macd_line, signal_line = compute_macd(pd.Series(closes), fast=12, slow=26, signal=9)
    ref_macd, ref_signal = _reference_macd(closes)

    np.testing.assert_allclose(macd_line.values, np.array(ref_macd), equal_nan=True, atol=1e-6)
    np.testing.assert_allclose(signal_line.values, np.array(ref_signal), equal_nan=True, atol=1e-6)
    assert macd_line.iloc[:25].isna().all()
    assert macd_line.iloc[25:].notna().all()


def test_macd_no_future_dependency():
    closes = [100 + 0.5 * i + 3 * np.sin(i / 4) for i in range(60)]
    macd_before, signal_before = compute_macd(pd.Series(closes))

    perturbed = list(closes)
    perturbed[55] += 1000
    macd_after, signal_after = compute_macd(pd.Series(perturbed))

    np.testing.assert_allclose(macd_before.iloc[:55].values, macd_after.iloc[:55].values, equal_nan=True)
    np.testing.assert_allclose(signal_before.iloc[:55].values, signal_after.iloc[:55].values, equal_nan=True)


# ---------------------------------------------------------------------------
# E. Bollinger Bands
# ---------------------------------------------------------------------------
def test_bollinger_matches_independent_reference_and_warmup():
    closes = [100 + (i % 7) - 3 for i in range(30)]
    upper, lower = compute_bollinger(pd.Series(closes), window=20, num_std=2.0)

    for t in range(19, 30):
        window = closes[t - 19 : t + 1]
        mean = np.mean(window)
        std = np.std(window, ddof=1)
        assert upper.iloc[t] == pytest.approx(mean + 2 * std, rel=1e-9)
        assert lower.iloc[t] == pytest.approx(mean - 2 * std, rel=1e-9)

    assert upper.iloc[:19].isna().all()
    assert lower.iloc[:19].isna().all()


def test_bollinger_no_future_dependency():
    closes = [100 + (i % 7) - 3 for i in range(30)]
    upper_before, lower_before = compute_bollinger(pd.Series(closes))

    perturbed = list(closes)
    perturbed[28] += 50
    upper_after, lower_after = compute_bollinger(pd.Series(perturbed))

    np.testing.assert_allclose(upper_before.iloc[:28].values, upper_after.iloc[:28].values, equal_nan=True)
    np.testing.assert_allclose(lower_before.iloc[:28].values, lower_after.iloc[:28].values, equal_nan=True)


# ---------------------------------------------------------------------------
# F. Volatility
# ---------------------------------------------------------------------------
def test_volatility_matches_independent_reference_and_warmup():
    closes = [100 * (1.001 ** i) * (1 + 0.01 * np.sin(i)) for i in range(30)]
    vol = compute_volatility(pd.Series(closes), window=20)

    returns = [np.nan] + [closes[i] / closes[i - 1] - 1 for i in range(1, len(closes))]
    for t in range(20, 30):
        window = returns[t - 19 : t + 1]
        expected = np.std(window, ddof=1)
        assert vol.iloc[t] == pytest.approx(expected, rel=1e-9)

    # Insufficient history (fewer than 20 return observations) -> NaN.
    assert vol.iloc[:20].isna().all()


def test_volatility_no_future_dependency():
    closes = [100 * (1.001 ** i) * (1 + 0.01 * np.sin(i)) for i in range(30)]
    before = compute_volatility(pd.Series(closes))

    perturbed = list(closes)
    perturbed[27] *= 1.5
    after = compute_volatility(pd.Series(perturbed))

    np.testing.assert_allclose(before.iloc[:27].values, after.iloc[:27].values, equal_nan=True)


# ---------------------------------------------------------------------------
# G. Momentum (generic — see the "Momentum lookbacks" decision gate)
# ---------------------------------------------------------------------------
def test_momentum_exact_offset_no_off_by_one():
    closes = list(range(100, 140))  # 100, 101, ..., 139
    lookback = 10
    mom = compute_momentum(pd.Series(closes), lookback_sessions=lookback)

    for t in range(lookback, len(closes)):
        expected = closes[t] / closes[t - lookback] - 1
        assert mom.iloc[t] == pytest.approx(expected)
    assert mom.iloc[:lookback].isna().all()


def test_momentum_no_future_dependency():
    closes = [100 + i + 2 * np.sin(i / 3) for i in range(40)]
    lookback = 15
    before = compute_momentum(pd.Series(closes), lookback_sessions=lookback)

    perturbed = list(closes)
    perturbed[35] += 200
    after = compute_momentum(pd.Series(perturbed), lookback_sessions=lookback)

    np.testing.assert_allclose(before.iloc[:35].values, after.iloc[:35].values, equal_nan=True)


def test_momentum_rejects_nonpositive_lookback():
    with pytest.raises(ValueError):
        compute_momentum(pd.Series([1.0, 2.0, 3.0]), lookback_sessions=0)


def test_technical_features_momentum_defaults_to_locked_config_constants():
    """The momentum decision gate is now LOCKED (config.MOMENTUM_3M_SESSIONS
    = 63, config.MOMENTUM_6M_SESSIONS = 126) — a default production call
    (no momentum_*_sessions passed) must compute real values, never NULL.
    See tests/test_momentum_locked.py for the full locking-test suite.
    """
    import config

    prices = _prices("AAPL", [100 + i for i in range(150)])
    out = compute_technical_features(prices)  # no momentum_*_sessions passed
    assert out["momentum_3m"].iloc[config.MOMENTUM_3M_SESSIONS :].notna().all()
    assert out["momentum_6m"].iloc[config.MOMENTUM_6M_SESSIONS :].notna().all()

    # An explicit override is still honored (generic parameterization is preserved).
    out2 = compute_technical_features(prices, momentum_3m_sessions=5)
    assert out2["momentum_3m"].notna().sum() > out["momentum_3m"].notna().sum()


# ---------------------------------------------------------------------------
# H. Macro alignment
# ---------------------------------------------------------------------------
def test_macro_alignment_by_date_not_row_position_no_future_no_fill():
    # Sessions on 3 dates; macro has irregular coverage (blank padding /
    # missing date -> that session's macro fields stay NULL, never filled).
    sessions = pd.DataFrame(
        {
            "ticker": ["AAPL", "AAPL", "AAPL"],
            "date": pd.to_datetime(["2021-01-04", "2021-01-05", "2021-01-06"]),
        }
    )
    macro = pd.DataFrame(
        {
            "date": pd.to_datetime(["2021-01-04", "2021-01-06", "2021-01-07"]),  # 01-05 missing, 01-07 is future
            "spx": [3800.0, 3820.0, 3900.0],
            "vix": [20.0, 21.0, 99.0],
            "yield_10y": [1.0, 1.1, 5.0],
        }
    )
    out = align_macro_to_sessions(sessions, macro)

    row_04 = out[out["date"] == "2021-01-04"].iloc[0]
    row_05 = out[out["date"] == "2021-01-05"].iloc[0]
    row_06 = out[out["date"] == "2021-01-06"].iloc[0]

    assert row_04["vix"] == 20.0 and row_04["yield_10y"] == 1.0
    # 01-05 has no exact macro date -> NULL, never nearest-filled from 01-04 or 01-06.
    assert pd.isna(row_05["vix"]) and pd.isna(row_05["yield_10y"])
    assert row_06["vix"] == 21.0 and row_06["yield_10y"] == 1.1
    # The 01-07 (future-dated) macro row must never appear anywhere in the output.
    assert "2021-01-07" not in out["date"].astype(str).values
    # Macro alignment must never create a new stock session.
    assert len(out) == len(sessions)


def test_macro_alignment_never_creates_stock_sessions():
    sessions = pd.DataFrame({"ticker": ["AAPL"], "date": pd.to_datetime(["2021-01-04"])})
    macro = pd.DataFrame(
        {
            "date": pd.to_datetime([f"2021-01-{d:02d}" for d in range(1, 10)]),
            "spx": list(range(9)),
            "vix": list(range(9)),
            "yield_10y": list(range(9)),
        }
    )
    out = align_macro_to_sessions(sessions, macro)
    assert len(out) == 1


def test_spx_return_no_future_dependency():
    macro = pd.DataFrame(
        {
            "date": pd.bdate_range("2021-01-04", periods=10),
            "spx": [3800 + 5 * i for i in range(10)],
        }
    )
    before = compute_spx_return(macro, horizon=1)
    perturbed = macro.copy()
    perturbed.loc[8, "spx"] = 999999.0
    after = compute_spx_return(perturbed, horizon=1)

    np.testing.assert_allclose(
        before["spx_return"].iloc[:8].values, after["spx_return"].iloc[:8].values, equal_nan=True
    )


# ---------------------------------------------------------------------------
# I. Missing data / J. Static leakage
# ---------------------------------------------------------------------------
def test_static_snapshot_never_leaks_into_persisted_feature_columns():
    prices = _prices("AAPL", [100 + i for i in range(30)])
    macro = pd.DataFrame(
        {
            "date": prices["date"],
            "vix": [20.0] * 30,
            "yield_10y": [1.5] * 30,
        }
    )
    out = build_feature_frame(prices, macro)

    for col in STATIC_LEAKAGE_COLS:
        assert out[col].isna().all(), f"{col} must be NULL for every Phase 3 persisted row"


def test_no_backward_fill_or_zero_fabrication_in_warmup_rows():
    prices = _prices("AAPL", [100 + i for i in range(10)])  # too short for any indicator's full window
    macro = pd.DataFrame({"date": prices["date"], "vix": [20.0] * 10, "yield_10y": [1.0] * 10})
    out = build_feature_frame(prices, macro)

    # Warm-up rows are genuinely NaN, never 0 or back-filled from a later row.
    assert out["rsi_14"].isna().all()  # only 10 sessions, RSI needs 14
    assert (out["rsi_14"] == 0).sum() == 0
    assert (out["bb_upper"] == 0).sum() == 0


# ---------------------------------------------------------------------------
# K. Ticker isolation
# ---------------------------------------------------------------------------
def test_ticker_isolation_interleaved_rows_no_cross_contamination():
    aapl = _prices("AAPL", [100 + i for i in range(40)])
    msft = _prices("MSFT", [500 - 2 * i for i in range(40)])  # deliberately different trend
    combined = pd.concat([aapl, msft]).sample(frac=1.0, random_state=3).reset_index(drop=True)

    out_combined = compute_technical_features(combined)
    out_aapl_alone = compute_technical_features(aapl)
    out_msft_alone = compute_technical_features(msft)

    for col in ["rsi_14", "macd", "bb_upper", "volatility_20d"]:
        a = out_combined[out_combined["ticker"] == "AAPL"][col].reset_index(drop=True)
        b = out_aapl_alone[col].reset_index(drop=True)
        np.testing.assert_allclose(a.values.astype(float), b.values.astype(float), equal_nan=True)

        m = out_combined[out_combined["ticker"] == "MSFT"][col].reset_index(drop=True)
        n = out_msft_alone[col].reset_index(drop=True)
        np.testing.assert_allclose(m.values.astype(float), n.values.astype(float), equal_nan=True)


# ---------------------------------------------------------------------------
# L. THE FUTURE-PERTURBATION CAUSALITY TEST (comprehensive, all features)
# ---------------------------------------------------------------------------
def test_future_perturbation_causality_all_features():
    """The single most important Phase 3 test: perturb every price/macro
    observation strictly after cutoff date t, and assert EVERY feature
    value at or before t is byte-for-byte identical. Covers RSI, MACD,
    MACD signal, Bollinger upper/lower, volatility, momentum (with an
    explicit lookback so the column is populated), VIX, yield_10y, and the
    (unpersisted) SPX-derived feature.
    """
    n = 80
    rng = np.random.RandomState(42)
    closes = list(100 + np.cumsum(rng.normal(0, 1, n)))
    aapl = _prices("AAPL", closes)
    msft = _prices("MSFT", list(200 + np.cumsum(rng.normal(0, 1, n))))
    prices = pd.concat([aapl, msft]).reset_index(drop=True)

    macro = pd.DataFrame(
        {
            "date": pd.bdate_range("2021-01-04", periods=n),
            "spx": list(3800 + np.cumsum(rng.normal(0, 5, n))),
            "vix": list(20 + rng.normal(0, 1, n)),
            "yield_10y": list(1.5 + rng.normal(0, 0.05, n)),
        }
    )

    cutoff = 50  # 0-indexed session position

    before = build_feature_frame(prices, macro, momentum_3m_sessions=10, momentum_6m_sessions=20)
    before_spx = compute_spx_return(macro)

    perturbed_prices = prices.copy()
    perturbed_prices.loc[perturbed_prices.groupby("ticker").cumcount() > cutoff, "close"] += 10_000
    perturbed_prices.loc[perturbed_prices.groupby("ticker").cumcount() > cutoff, "total_return_idx"] += 10_000
    perturbed_macro = macro.copy()
    perturbed_macro.loc[cutoff + 1 :, ["spx", "vix", "yield_10y"]] += 10_000

    after = build_feature_frame(
        perturbed_prices, perturbed_macro, momentum_3m_sessions=10, momentum_6m_sessions=20
    )
    after_spx = compute_spx_return(perturbed_macro)

    feature_cols = [
        "rsi_14", "macd", "macd_signal", "bb_upper", "bb_lower",
        "volatility_20d", "momentum_3m", "momentum_6m", "vix", "yield_10y",
    ]
    for ticker in ("AAPL", "MSFT"):
        b = before[before["ticker"] == ticker].reset_index(drop=True)
        a = after[after["ticker"] == ticker].reset_index(drop=True)
        for col in feature_cols:
            np.testing.assert_allclose(
                b[col].iloc[: cutoff + 1].values.astype(float),
                a[col].iloc[: cutoff + 1].values.astype(float),
                equal_nan=True,
                err_msg=f"{ticker}.{col} changed at/before cutoff after a future-only perturbation",
            )

    np.testing.assert_allclose(
        before_spx["spx_return"].iloc[: cutoff + 1].values.astype(float),
        after_spx["spx_return"].iloc[: cutoff + 1].values.astype(float),
        equal_nan=True,
    )


# ---------------------------------------------------------------------------
# N. Determinism
# ---------------------------------------------------------------------------
def test_determinism_shuffled_input_same_sorted_output():
    aapl = _prices("AAPL", [100 + i + (i % 5) for i in range(50)])
    macro = pd.DataFrame({"date": aapl["date"], "vix": [20.0] * 50, "yield_10y": [1.5] * 50})

    out1 = build_feature_frame(aapl, macro, momentum_3m_sessions=5)
    shuffled = aapl.sample(frac=1.0, random_state=11).reset_index(drop=True)
    out2 = build_feature_frame(shuffled, macro, momentum_3m_sessions=5)

    pd.testing.assert_frame_equal(out1.reset_index(drop=True), out2.reset_index(drop=True), check_exact=False)


def test_build_feature_frame_one_row_per_ticker_date_no_duplicates():
    aapl = _prices("AAPL", [100 + i for i in range(30)])
    macro = pd.DataFrame({"date": aapl["date"], "vix": [20.0] * 30, "yield_10y": [1.5] * 30})
    out = build_feature_frame(aapl, macro)
    assert not out.duplicated(subset=["ticker", "date"]).any()
    assert len(out) == len(valid_sessions(aapl))
