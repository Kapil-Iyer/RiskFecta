"""
RiskFecta Phase 3 — leakage-safe technical/macro feature computation and
`features`-table persistence.

Scope (locked, see BUILD_PLAN.md Phase 3, ML_SPEC.md §5-§8, §10, and the
Phase 3 task brief):

  - Every feature at (ticker, date=t) uses only information available at or
    before t — backward-looking rolling/EWM windows only, computed strictly
    per ticker (pipeline.sessions.valid_sessions + a per-ticker groupby;
    never a global/cross-ticker rolling calculation).
  - Static snapshot fields (static_fields.csv: market cap, beta, dividend
    yield, sector) are a single current-day Bloomberg snapshot, not
    point-in-time history (ML_SPEC.md §10). They are NEVER attached to
    historical (ticker, date) feature rows here — `beta`, `mkt_cap_log`,
    `sector`, `div_yield` are always written as SQL NULL by this module.
  - No `schema.sql` change is made or required by this module. It writes
    only to the existing `features` table, using only columns the frozen
    schema already defines.

Three Phase 3 decision gates, all now LOCKED (resolved explicitly by the
user after the Phase 3 pre-audit report and Cursor's conditional-pass
review — see README.md "Phase 3 methodology decisions (locked)" for the
discoverable record):

  1. Momentum lookbacks — LOCKED. ML_SPEC.md §6/§15 name "momentum
     (3-month, 6-month equivalent in trading sessions)" without freezing an
     exact session count. RiskFecta V2 Phase 3 methodology now locks this
     to the standard ~21-trading-sessions/month approximation:
     `config.MOMENTUM_3M_SESSIONS = 63`, `config.MOMENTUM_6M_SESSIONS = 126`
     — defined ONLY in config.py, never hardcoded independently here.
     `compute_momentum()` remains a generic, independently parameterized/
     testable function; `compute_technical_features()`/`build_feature_frame()`
     default `momentum_3m_sessions`/`momentum_6m_sessions` to these config
     constants when the caller does not explicitly override them, so
     production Phase 3 feature generation always uses 63/126 unless a
     future, explicitly approved methodology revision changes it.
  2. Macro forward-fill horizon — LOCKED as "no fill." ML_SPEC.md §5
     explicitly flags "the exact limit is a Phase 3 decision gate ... not
     fabricated here," and no maximum forward-fill horizon is defined
     anywhere in the current V2 docs. The locked MVP policy is EXACT-DATE
     alignment only: `align_macro_to_sessions()` performs only an exact-
     date, same-day join (never bfill, never nearest/asof, never a
     row-position join) — a macro value not quoted exactly on a session's
     own date is left NULL. This is the conservative, leakage-safe MVP
     policy itself (not a placeholder pending a fill-horizon number); a
     future explicit methodology decision would be required to change it.
  3. SPX persistence — LOCKED as "in-memory only, not persisted." ML_SPEC.md
     §6 names "SPX level/return" as a macro feature family member, but
     schema.sql has no `spx`/`spx_return` column and neither
     `config.XGBOOST_FEATURE_COLS` nor `config.LSTM_FEATURE_COLS` requires
     one. `compute_spx_return()` implements date-safe SPX alignment/return
     as an in-memory-only computation, kept and tested, but deliberately
     NOT part of `build_feature_frame()`'s persisted columns and NOT
     written to `features` — no schema/config change is made to add it.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from psycopg2.extras import execute_values

import config
from pipeline import db
from pipeline.sessions import DATE_COL, TICKER_COL, valid_sessions

# ---------------------------------------------------------------------------
# Column contracts
# ---------------------------------------------------------------------------
TECHNICAL_COLS = [
    "rsi_14", "macd", "macd_signal", "bb_upper", "bb_lower",
    "volatility_20d", "momentum_3m", "momentum_6m",
]
MACRO_COLS = ["vix", "yield_10y"]
# Order matches schema.sql's `features` table column order exactly.
STATIC_LEAKAGE_COLS = ["beta", "mkt_cap_log", "sector", "div_yield"]
FEATURES_TABLE_COLUMNS = (
    [TICKER_COL, DATE_COL] + TECHNICAL_COLS + STATIC_LEAKAGE_COLS + MACRO_COLS
)


# ---------------------------------------------------------------------------
# Technical indicators — each operates on a single ticker's own close
# series, ordered ascending by date (callers apply these per-ticker; see
# compute_technical_features). Every one is backward-looking only: at
# position t it reads close[0..t], never close[t+1:].
# ---------------------------------------------------------------------------
def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Wilder-style RSI(period): average gain/loss via a recursive EWM
    (alpha = 1/period, no SMA seed — same "pure recursive from the first
    observation" convention used for MACD below; see module docstring and
    the Phase 3 pre-audit report). The first `period` positions are
    explicitly masked to NaN (warm-up: fewer than `period` price changes
    exist yet) regardless of the EWM's own internal behavior.
    """
    close = close.reset_index(drop=True)
    delta = close.diff()
    gain = delta.clip(lower=0).fillna(0.0)
    loss = (-delta.clip(upper=0)).fillna(0.0)

    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False).mean()

    with np.errstate(divide="ignore", invalid="ignore"):
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
    rsi = rsi.where(avg_loss != 0, 100.0)  # no losses in the lookback -> RSI 100, not inf/NaN
    rsi.iloc[: min(period, len(rsi))] = np.nan
    return rsi


def compute_macd(
    close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
) -> Tuple[pd.Series, pd.Series]:
    """MACD(fast=12, slow=26, signal=9): EMA(fast) - EMA(slow), and its own
    EMA(signal). EMAs use pandas' pure recursive form (`adjust=False`, no
    SMA seed) — an explicit implementation choice documented here and in
    the Phase 3 pre-audit report (ML_SPEC.md does not pin SMA-seed vs.
    pure-recursive EMA). Warm-up: `macd` is masked NaN before `slow`
    observations exist; `macd_signal` is masked NaN before
    `slow + signal - 1` observations exist.
    """
    close = close.reset_index(drop=True)
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    macd_line.iloc[: min(slow - 1, len(macd_line))] = np.nan

    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    warm = min(slow - 1 + signal - 1, len(signal_line))
    signal_line.iloc[:warm] = np.nan
    return macd_line, signal_line


def compute_bollinger(
    close: pd.Series, window: int = 20, num_std: float = 2.0
) -> Tuple[pd.Series, pd.Series]:
    """Bollinger Bands: SMA(window) ± num_std * trailing sample stdev
    (ddof=1) of close, over the trailing `window` sessions ending at t
    (never centered). `rolling(..., min_periods=window)` naturally NaNs the
    first `window - 1` positions (insufficient history).
    """
    close = close.reset_index(drop=True)
    sma = close.rolling(window=window, min_periods=window).mean()
    std = close.rolling(window=window, min_periods=window).std(ddof=1)
    upper = sma + num_std * std
    lower = sma - num_std * std
    return upper, lower


def compute_volatility(close: pd.Series, window: int = 20) -> pd.Series:
    """Trailing realized volatility: sample stdev (ddof=1) of simple daily
    close-to-close returns over the trailing `window` sessions ending at t.
    Not annualized — ML_SPEC.md §6/§8 does not specify an annualization
    factor for this feature, so none is applied here (documented choice;
    see the Phase 3 pre-audit report). Never a centered window.
    """
    close = close.reset_index(drop=True)
    returns = close.pct_change()
    return returns.rolling(window=window, min_periods=window).std(ddof=1)


def compute_momentum(close: pd.Series, lookback_sessions: int) -> pd.Series:
    """Price momentum over an explicit number of valid trading sessions:
    close[t] / close[t - lookback_sessions] - 1. Purely backward-looking
    (`shift` only ever looks at earlier positions) and fully general —
    remains independently parameterized/testable with any lookback. The
    locked production 3-month/6-month session counts
    (`config.MOMENTUM_3M_SESSIONS` / `config.MOMENTUM_6M_SESSIONS`, see
    module docstring) are wired in by `compute_technical_features()` /
    `build_feature_frame()` below, not hardcoded in this function.
    """
    if lookback_sessions <= 0:
        raise ValueError("compute_momentum: lookback_sessions must be a positive valid-session count")
    close = close.reset_index(drop=True)
    return close / close.shift(lookback_sessions) - 1


# ---------------------------------------------------------------------------
# Macro alignment — date-keyed, never row-position-keyed (ML_SPEC.md §29).
# ---------------------------------------------------------------------------
def align_macro_to_sessions(sessions: pd.DataFrame, macro: pd.DataFrame) -> pd.DataFrame:
    """Left-join VIX/yield_10y onto each (ticker, date) valid session by
    exact calendar date (macro's own `date`/`vix`/`yield_10y` columns — see
    pipeline.normalize.normalize_macro). `sessions` must already be
    valid-session-filtered (pipeline.sessions.valid_sessions); this
    function does not gate on any price field itself and does not create,
    drop, or reorder stock sessions.

    No forward-fill / no nearest-date / no row-position join: a macro
    value not quoted exactly on a session's date is left NULL (see the
    "Macro forward-fill horizon" decision gate in the module docstring).
    This can never pull a macro observation dated after the session's date
    into that row — an exact-date join has no such path.
    """
    missing = [c for c in ("vix", "yield_10y") if c not in macro.columns]
    if missing:
        raise ValueError(f"align_macro_to_sessions: macro frame missing column(s) {missing}")
    return sessions[[TICKER_COL, DATE_COL]].merge(
        macro[[DATE_COL, "vix", "yield_10y"]], on=DATE_COL, how="left"
    )


def compute_spx_return(macro: pd.DataFrame, horizon: int = 1) -> pd.DataFrame:
    """Date-safe SPX level + trailing `horizon`-session return, computed
    independently of any stock's own session calendar (SPX has its own
    valid-quote dates — pipeline.validate.parse_macro_raw). Backward-looking
    only: spx_return(t) = spx(t)/spx(t-horizon) - 1, using only spx(t) and
    earlier observations.

    NOT wired into build_feature_frame()/persisted `features` rows — see
    module docstring (SPX persistence decision gate). Exposed here purely
    as an in-memory, independently-testable computation for later,
    explicitly authorized use once a persistence destination is decided.
    """
    if "spx" not in macro.columns:
        raise ValueError("compute_spx_return: macro frame missing 'spx' column")
    df = macro[[DATE_COL, "spx"]].dropna(subset=["spx"]).sort_values(DATE_COL).reset_index(drop=True)
    df["spx_return"] = df["spx"] / df["spx"].shift(horizon) - 1
    return df


# ---------------------------------------------------------------------------
# Full technical-feature surface (per-ticker, ticker-isolated)
# ---------------------------------------------------------------------------
def compute_technical_features(
    prices: pd.DataFrame,
    momentum_3m_sessions: Optional[int] = None,
    momentum_6m_sessions: Optional[int] = None,
) -> pd.DataFrame:
    """RSI/MACD/Bollinger/volatility/momentum for every valid session of
    every ticker in `prices`, computed strictly per ticker: each ticker's
    rolling/EWM state is reset at that ticker's own first valid session, so
    no ticker's calculation can ever read another ticker's rows (tested
    explicitly — see tests/test_features.py ticker-isolation cases).

    momentum_3m_sessions / momentum_6m_sessions: default to the LOCKED
    production values config.MOMENTUM_3M_SESSIONS (63) /
    config.MOMENTUM_6M_SESSIONS (126) when not explicitly overridden - see
    the "Momentum lookbacks" decision in the module docstring. An explicit
    override (e.g. a test exercising a different lookback) is still
    honored; the momentum decision gate is resolved, not optional, so a
    default production call never leaves these columns NULL.
    """
    momentum_3m_sessions = (
        config.MOMENTUM_3M_SESSIONS if momentum_3m_sessions is None else momentum_3m_sessions
    )
    momentum_6m_sessions = (
        config.MOMENTUM_6M_SESSIONS if momentum_6m_sessions is None else momentum_6m_sessions
    )

    sess = valid_sessions(prices)
    frames: List[pd.DataFrame] = []
    for ticker, g in sess.groupby(TICKER_COL, sort=False):
        g = g.sort_values(DATE_COL, kind="mergesort").reset_index(drop=True)
        close = g["close"]

        rsi = compute_rsi(close)
        macd_line, macd_signal = compute_macd(close)
        bb_upper, bb_lower = compute_bollinger(close)
        vol = compute_volatility(close)
        mom3 = (
            compute_momentum(close, momentum_3m_sessions)
            if momentum_3m_sessions
            else pd.Series(np.nan, index=close.index)
        )
        mom6 = (
            compute_momentum(close, momentum_6m_sessions)
            if momentum_6m_sessions
            else pd.Series(np.nan, index=close.index)
        )

        frames.append(
            pd.DataFrame(
                {
                    TICKER_COL: ticker,
                    DATE_COL: g[DATE_COL].values,
                    "rsi_14": rsi.values,
                    "macd": macd_line.values,
                    "macd_signal": macd_signal.values,
                    "bb_upper": bb_upper.values,
                    "bb_lower": bb_lower.values,
                    "volatility_20d": vol.values,
                    "momentum_3m": mom3.values,
                    "momentum_6m": mom6.values,
                }
            )
        )

    if not frames:
        return pd.DataFrame(columns=[TICKER_COL, DATE_COL] + TECHNICAL_COLS)

    result = pd.concat(frames, ignore_index=True)
    return result.sort_values([TICKER_COL, DATE_COL], kind="mergesort").reset_index(drop=True)


def build_feature_frame(
    prices: pd.DataFrame,
    macro: pd.DataFrame,
    momentum_3m_sessions: Optional[int] = None,
    momentum_6m_sessions: Optional[int] = None,
) -> pd.DataFrame:
    """Full `features`-table-shaped frame: technical features + macro
    (vix, yield_10y) alignment + static-snapshot columns forced to SQL NULL
    (beta, mkt_cap_log, sector, div_yield — ML_SPEC.md §10; the current-day
    static_fields.csv snapshot is never attached to these historical rows).

    This is the production Phase 3 orchestration entry point: called with
    no momentum arguments, it uses the LOCKED config.MOMENTUM_3M_SESSIONS
    (63) / config.MOMENTUM_6M_SESSIONS (126) values (via
    compute_technical_features's own default resolution) — momentum_3m/
    momentum_6m are never left NULL in a default production call.

    One row per (ticker, date); raises if that invariant is ever violated.
    """
    tech = compute_technical_features(prices, momentum_3m_sessions, momentum_6m_sessions)
    sess = valid_sessions(prices)[[TICKER_COL, DATE_COL]]
    macro_aligned = align_macro_to_sessions(sess, macro)

    out = tech.merge(macro_aligned, on=[TICKER_COL, DATE_COL], how="left")
    for col in STATIC_LEAKAGE_COLS:
        out[col] = None
    out = out[FEATURES_TABLE_COLUMNS]

    if out.duplicated(subset=[TICKER_COL, DATE_COL]).any():
        raise ValueError("build_feature_frame: duplicate (ticker, date) rows produced")
    return out.sort_values([TICKER_COL, DATE_COL], kind="mergesort").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Persistence — `features` table only. Idempotent UPSERT keyed on the
# table's own UNIQUE(ticker, date), mirroring pipeline.ingest.upsert_prices.
# Never touches prices_raw, predictions, portfolios, or risk_metrics.
# ---------------------------------------------------------------------------
FEATURES_UPSERT_SQL = """
INSERT INTO features (
    ticker, date, rsi_14, macd, macd_signal, bb_upper, bb_lower,
    volatility_20d, momentum_3m, momentum_6m, beta, mkt_cap_log, sector,
    div_yield, vix, yield_10y
) VALUES %s
ON CONFLICT (ticker, date) DO UPDATE SET
    rsi_14 = EXCLUDED.rsi_14,
    macd = EXCLUDED.macd,
    macd_signal = EXCLUDED.macd_signal,
    bb_upper = EXCLUDED.bb_upper,
    bb_lower = EXCLUDED.bb_lower,
    volatility_20d = EXCLUDED.volatility_20d,
    momentum_3m = EXCLUDED.momentum_3m,
    momentum_6m = EXCLUDED.momentum_6m,
    beta = EXCLUDED.beta,
    mkt_cap_log = EXCLUDED.mkt_cap_log,
    sector = EXCLUDED.sector,
    div_yield = EXCLUDED.div_yield,
    vix = EXCLUDED.vix,
    yield_10y = EXCLUDED.yield_10y
"""


def _to_sql_value(v):
    """NaN/NaT/None -> Python None (SQL NULL); everything else passed
    through untouched. Never fabricates a non-NULL value for a genuinely
    missing one.
    """
    if v is None:
        return None
    if isinstance(v, float) and np.isnan(v):
        return None
    if v is pd.NaT:
        return None
    try:
        if pd.isna(v):
            return None
    except (TypeError, ValueError):
        pass
    return v


def _prepare_features_for_insert(df: pd.DataFrame) -> List[Tuple]:
    """features-table-shaped DataFrame -> list of DB-ready row tuples, in
    FEATURES_TABLE_COLUMNS order, with every NaN/None normalized to Python
    None (-> SQL NULL) and `date` normalized to `datetime.date`.
    """
    out = df[FEATURES_TABLE_COLUMNS].copy()
    out[DATE_COL] = pd.to_datetime(out[DATE_COL]).dt.date
    records = []
    for row in out.itertuples(index=False, name=None):
        records.append(tuple(_to_sql_value(v) for v in row))
    return records


def upsert_features(conn, df: pd.DataFrame) -> int:
    """Low-level UPSERT of an already-built features-table-shaped
    DataFrame into `features`. Does NOT commit — caller controls the
    transaction boundary (see persist_features).
    """
    records = _prepare_features_for_insert(df)
    if not records:
        return 0
    with conn.cursor() as cur:
        execute_values(cur, FEATURES_UPSERT_SQL, records, page_size=1000)
    return len(records)


@dataclass
class PersistResult:
    rows_prepared: int
    rows_in_table_after: int


def persist_features(conn=None, df: Optional[pd.DataFrame] = None) -> PersistResult:
    """Ingest an already-built features DataFrame into the `features`
    table. Idempotent (UNIQUE(ticker, date) + ON CONFLICT DO UPDATE);
    transactional (any failure rolls back the whole batch, no partial
    writes); touches only `features` — never prices_raw, predictions,
    portfolios, or risk_metrics.

    IMPORTANT (Phase 3 Integrity Audit gate): this function is implemented
    and unit/mock-tested in Phase 3, but is NOT called against the real,
    live database with real Bloomberg-derived feature rows until the
    independent Cursor Integrity Audit has passed and the user has
    explicitly approved real execution (see the Phase 3 pre-audit report).
    """
    if df is None:
        raise ValueError("persist_features: df is required (no default real feature computation here)")
    owns_conn = conn is None
    conn = conn or db.get_connection()
    try:
        with conn:
            n = upsert_features(conn, df)
        with conn:
            rows_after = db.fetch_scalar(conn, "SELECT COUNT(*) FROM features")
        return PersistResult(rows_prepared=n, rows_in_table_after=int(rows_after))
    finally:
        if owns_conn:
            conn.close()
