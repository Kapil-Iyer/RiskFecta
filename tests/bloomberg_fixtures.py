"""
Small, hand-built Bloomberg-shaped CSV fixtures for Phase 1A tests.

These deliberately mimic the real export's quirks confirmed during the
Phase 1A audit (stray column 0 in prices_raw.csv, three independent
date/value column pairs in macro.csv) at a tiny scale, so tests never touch
the real, gitignored data/raw/ files (TRD.md §17).
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence

FIELD_ORDER = ["PX_OPEN", "PX_HIGH", "PX_LOW", "PX_LAST", "PX_VOLUME", "TOTAL_RETURN_INDEX"]


def _fmt(v) -> str:
    if v is None:
        return "#N/A"
    return str(v)


def write_prices_fixture(
    path: Path,
    tickers: Sequence[str],
    rows: List[Dict[str, object]],
    field_order: Optional[Sequence[str]] = None,
    corrupt_field_order_for: Optional[str] = None,
    drop_last_column: bool = False,
) -> None:
    """Write a small Bloomberg-shaped wide prices CSV.

    ``rows`` is a list of dicts: {"date": "YYYY-MM-DD", TICKER: {field: value, ...}, ...}.
    A ticker/date combination not given a field dict is written as all-#N/A
    with TOTAL_RETURN_INDEX defaulting to "1" (mirrors the real export's
    weekend/holiday placeholder rows).
    """
    field_order = list(field_order) if field_order is not None else FIELD_ORDER
    lines = []

    # Row 0: stray column 0 (unrelated ticker-list artifact) + ticker header blocks.
    header0 = [f"{tickers[0]} US Equity", ""]
    for t in tickers:
        block_fields = field_order
        if corrupt_field_order_for == t:
            block_fields = list(reversed(field_order))
        header0 += [f"{t} US Equity"] * len(block_fields)
    lines.append(",".join(header0))

    # Row 1: stray column 0 + "DATES" + field names per ticker block.
    header1 = [f"{tickers[-1]} US Equity", "DATES"]
    for t in tickers:
        block_fields = field_order
        if corrupt_field_order_for == t:
            block_fields = list(reversed(field_order))
        header1 += list(block_fields)
    lines.append(",".join(header1))

    for i, row in enumerate(rows):
        line = [f"{tickers[i % len(tickers)]} US Equity", row["date"]]
        for t in tickers:
            values = row.get(t)
            if values is None:
                line += ["#N/A"] * (len(field_order) - 1) + ["1"]
            else:
                line += [_fmt(values.get(f)) for f in field_order]
        lines.append(",".join(line))

    text = "\n".join(lines) + "\n"
    if drop_last_column:
        # Strip the final column from every line (malformed-column-count case).
        text = "\n".join(",".join(l.split(",")[:-1]) for l in text.splitlines()) + "\n"
    path.write_text(text)


def write_macro_fixture(
    path: Path,
    spx: List[Dict[str, object]],
    vix: List[Dict[str, object]],
    usgg10yr: List[Dict[str, object]],
) -> None:
    """Write a small Bloomberg-shaped macro CSV with three independent,
    possibly different-length (date, value) series, matching the real
    export's column layout: SPX_DATE, SPX_PX_LAST, VIX_DATE, VIX_PX_LAST,
    USGG10YR_DATE, USGG10YR_PX_LAST.
    """
    n = max(len(spx), len(vix), len(usgg10yr))

    def cell(series: List[Dict[str, object]], i: int, key: str) -> str:
        if i >= len(series):
            return ""
        return _fmt(series[i][key])

    lines = ["SPX_DATE,SPX_PX_LAST,VIX_DATE,VIX_PX_LAST,USGG10YR_DATE,USGG10YR_PX_LAST"]
    for i in range(n):
        row = [
            cell(spx, i, "date"), cell(spx, i, "value"),
            cell(vix, i, "date"), cell(vix, i, "value"),
            cell(usgg10yr, i, "date"), cell(usgg10yr, i, "value"),
        ]
        lines.append(",".join(row))
    path.write_text("\n".join(lines) + "\n")


def write_static_fields_fixture(path: Path, rows: List[Dict[str, object]]) -> None:
    """rows: list of {"ticker": "AAPL", "CUR_MKT_CAP": ..., "BETA_RAW_OVERRIDABLE": ...,
    "DIVIDEND_INDICATED_YIELD": ..., "GICS_SECTOR_NAME": ...}
    """
    lines = [",CUR_MKT_CAP,BETA_RAW_OVERRIDABLE,DIVIDEND_INDICATED_YIELD,GICS_SECTOR_NAME"]
    for row in rows:
        lines.append(
            ",".join(
                [
                    f"{row['ticker']} US Equity",
                    _fmt(row.get("CUR_MKT_CAP")),
                    _fmt(row.get("BETA_RAW_OVERRIDABLE")),
                    _fmt(row.get("DIVIDEND_INDICATED_YIELD")),
                    str(row.get("GICS_SECTOR_NAME", "")),
                ]
            )
        )
    path.write_text("\n".join(lines) + "\n")
