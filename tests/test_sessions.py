"""
RiskFecta Phase 3 — session-validity gate tests (pipeline/sessions.py).

Covers task brief §18-B (session validity): PX_LAST gates, TOTAL_RETURN_INDEX
never gates, weekend/holiday placeholders excluded, ascending sort within
ticker, ticker boundaries respected.
"""
from __future__ import annotations

import pandas as pd

from pipeline.sessions import session_number, valid_sessions


def _rows():
    # AAPL: 2021-01-01 valid, 2021-01-02 weekend placeholder (close NaN,
    # total_return_idx still populated — must NOT become a session), 2021-01-03 valid.
    # MSFT: given out of order and interleaved with AAPL to check sort + isolation.
    return pd.DataFrame(
        [
            {"ticker": "AAPL", "date": "2021-01-01", "close": 100.0, "total_return_idx": 100.0},
            {"ticker": "MSFT", "date": "2021-01-01", "close": 200.0, "total_return_idx": 200.0},
            {"ticker": "AAPL", "date": "2021-01-02", "close": None, "total_return_idx": 100.0},  # weekend
            {"ticker": "AAPL", "date": "2021-01-03", "close": 101.0, "total_return_idx": 101.0},
            {"ticker": "MSFT", "date": "2021-01-02", "close": 201.0, "total_return_idx": 201.0},
        ]
    )


def test_px_last_gates_total_return_index_never_gates():
    df = _rows()
    df["date"] = pd.to_datetime(df["date"])
    out = valid_sessions(df)

    # The weekend AAPL row (close NaN, TRI populated) must be excluded.
    aapl_dates = out.loc[out["ticker"] == "AAPL", "date"].dt.strftime("%Y-%m-%d").tolist()
    assert aapl_dates == ["2021-01-01", "2021-01-03"]
    assert "2021-01-02" not in aapl_dates


def test_sorted_ascending_within_ticker_and_ticker_boundaries_respected():
    df = _rows()
    df["date"] = pd.to_datetime(df["date"])
    out = valid_sessions(df)

    for ticker, g in out.groupby("ticker"):
        assert g["date"].is_monotonic_increasing

    msft_dates = out.loc[out["ticker"] == "MSFT", "date"].dt.strftime("%Y-%m-%d").tolist()
    assert msft_dates == ["2021-01-01", "2021-01-02"]


def test_session_number_is_per_ticker_zero_based():
    df = _rows()
    df["date"] = pd.to_datetime(df["date"])
    out = valid_sessions(df)
    out["session_no"] = session_number(out)

    aapl = out[out["ticker"] == "AAPL"].reset_index(drop=True)
    assert aapl["session_no"].tolist() == [0, 1]
    msft = out[out["ticker"] == "MSFT"].reset_index(drop=True)
    assert msft["session_no"].tolist() == [0, 1]
