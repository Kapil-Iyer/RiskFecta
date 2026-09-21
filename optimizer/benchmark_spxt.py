"""
RiskFecta Phase 7B SPXT amendment — official S&P 500 Total Return (SPXT)
market benchmark (optimizer/benchmark_spxt.py).

Bloomberg security: `SPXT Index`. Bloomberg field: `PX_LAST`. SPXT is the
S&P 500 TOTAL RETURN index (dividends reinvested) — this is NOT the
`SPX_PX_LAST` price-only series already in `data/raw/macro.csv`
(`pipeline.normalize`'s `spx` column), which remains valid historical
macro data but was never total-return and is not this benchmark (see
ML_SPEC.md §2/§27 and the Phase 7B SPXT amendment report for the full
reconciliation).

Raw artifact: `data/raw/spxt_benchmark.csv` (immutable, gitignored,
obtained specifically for Phase 7, BEFORE any Phase 7 execution — no
Phase 7 portfolio result was ever observed). This module only ever reads
it; it never writes to `data/raw/` and never modifies the frozen
`prices_raw.csv`/`macro.csv`/`static_fields.csv` trio.

EVALUATION-ONLY / STRUCTURAL SEPARATION FROM CONSTRUCTION: nothing in
this module is imported by `optimizer/covariance.py`, `optimizer/portfolio.py`,
or `construct_portfolios_at()` in `optimizer/walkforward.py` — portfolio
construction is identical whether this module exists or not. This module
instead depends on `optimizer.walkforward`'s column-name constants (a
one-directional dependency); see
`tests/test_optimizer_spxt.py`'s structural checks.

Formula (§5, official): for the already-frozen Phase 7 (formation_date,
target_date) pair,

    spxt_total_return_21 = SPXT_PX_LAST(target_date) / SPXT_PX_LAST(formation_date) - 1

using the EXACT same T/T+21 endpoints as the portfolio experiment — never
an independently shifted date, never a nearest-date lookup, never
interpolation, never forward/backward fill. A missing exact endpoint
fails loudly.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

import config

SPXT_RAW_PATH = config.DATA_RAW / "spxt_benchmark.csv"

# Deployment-location fallbacks ONLY — the parse/validation logic below is
# identical no matter which path resolves. `data/raw/` is gitignored (private
# Bloomberg export; see BUILD_PLAN.md), so it is never present on a
# git-based Render deploy. `SPXT_BENCHMARK_PATH` lets an operator point at
# wherever the file was placed (e.g. a Render persistent disk);
# `/etc/secrets/spxt_benchmark.csv` is Render's own "Secret Files" mount
# convention for uploading a private file outside of git. Neither path is
# ever written to, and a local checkout with `data/raw/spxt_benchmark.csv`
# present behaves exactly as before (first candidate wins).
_RENDER_SECRET_PATH = Path("/etc/secrets/spxt_benchmark.csv")

SPXT_DATE_COL = "date"
SPXT_LEVEL_COL = "spxt_px_last"

_RAW_DATE_COL = "SPXT_DATE"
_RAW_LEVEL_COL = "SPXT_PX_LAST"


def _resolve_spxt_path() -> Path:
    if SPXT_RAW_PATH.exists():
        return SPXT_RAW_PATH
    env_override = os.environ.get("SPXT_BENCHMARK_PATH")
    if env_override and Path(env_override).exists():
        return Path(env_override)
    if _RENDER_SECRET_PATH.exists():
        return _RENDER_SECRET_PATH
    return SPXT_RAW_PATH  # none found — preserve the original not-found error path/message


def load_spxt_raw(path: Optional[Path] = None) -> pd.DataFrame:
    """Read-only parse of the immutable raw SPXT artifact (never writes
    to `path`). Fails loudly on: a missing expected raw column, a
    duplicate date, or any non-numeric/non-finite/non-positive
    `SPXT_PX_LAST` value — never silently drops or coerces a bad row."""
    path = _resolve_spxt_path() if path is None else path
    raw = pd.read_csv(path)

    missing_cols = [c for c in (_RAW_DATE_COL, _RAW_LEVEL_COL) if c not in raw.columns]
    if missing_cols:
        raise ValueError(f"load_spxt_raw: missing expected column(s) {missing_cols} in {path}")

    out = pd.DataFrame({
        SPXT_DATE_COL: pd.to_datetime(raw[_RAW_DATE_COL]),
        SPXT_LEVEL_COL: pd.to_numeric(raw[_RAW_LEVEL_COL], errors="coerce"),
    })

    dup_mask = out.duplicated(subset=[SPXT_DATE_COL], keep=False)
    if dup_mask.any():
        n = int(out.duplicated(subset=[SPXT_DATE_COL]).sum())
        raise ValueError(f"load_spxt_raw: {n} duplicate date(s) in {path}")

    bad = out[SPXT_LEVEL_COL].isna() | ~np.isfinite(out[SPXT_LEVEL_COL]) | (out[SPXT_LEVEL_COL] <= 0)
    if bad.any():
        raise ValueError(
            f"load_spxt_raw: {int(bad.sum())} non-finite/non-positive/non-numeric "
            f"SPXT_PX_LAST value(s) in {path}"
        )

    return out.sort_values(SPXT_DATE_COL).reset_index(drop=True)


def spxt_total_return(spxt: pd.DataFrame, formation_date, target_date) -> float:
    """spxt_total_return_21 = SPXT(target_date)/SPXT(formation_date) - 1
    (official, §5). Exact-date lookup only: raises if either endpoint is
    absent from `spxt` rather than substituting a nearest date,
    interpolating, or forward/backward filling."""
    levels = spxt.set_index(SPXT_DATE_COL)[SPXT_LEVEL_COL]
    formation_date = pd.Timestamp(formation_date)
    target_date = pd.Timestamp(target_date)
    if formation_date not in levels.index:
        raise ValueError(
            f"spxt_total_return: formation_date {formation_date.date()} has no exact SPXT quote "
            "(no nearest-date fallback)"
        )
    if target_date not in levels.index:
        raise ValueError(
            f"spxt_total_return: target_date {target_date.date()} has no exact SPXT quote "
            "(no nearest-date fallback)"
        )
    return float(levels.loc[target_date] / levels.loc[formation_date] - 1.0)


def verify_spxt_covers_formation_calendar(spxt: pd.DataFrame, calendar_df: pd.DataFrame) -> None:
    """Verify EVERY exact required formation_date/target_date endpoint in
    `calendar_df` (`optimizer.walkforward.load_formation_calendar` output,
    or an equivalent frame carrying those two columns) has an exact SPXT
    quote. Fails loudly, naming every missing date, rather than silently
    evaluating only the dates that happen to be covered."""
    from optimizer.walkforward import FORECAST_DATE_COL, TARGET_DATE_COL

    levels_index = set(pd.to_datetime(spxt[SPXT_DATE_COL]))
    required = set(pd.to_datetime(calendar_df[FORECAST_DATE_COL])) | set(pd.to_datetime(calendar_df[TARGET_DATE_COL]))
    missing = sorted(d for d in required if d not in levels_index)
    if missing:
        shown = [d.date().isoformat() for d in missing[:10]]
        raise ValueError(
            f"verify_spxt_covers_formation_calendar: {len(missing)} required formation/target date(s) "
            f"missing an exact SPXT quote: {shown}" + (" ..." if len(missing) > 10 else "")
        )
