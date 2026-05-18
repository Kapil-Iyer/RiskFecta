# Bloomberg Terminal Export Spec (Phase 1 reference)

**Data authenticity cannot be fake.** All pipeline data must come from Bloomberg export.

## Pull 1 — Prices (daily, 5 years)

- **Source:** BQL or Excel Bloomberg add-in, multi-asset panel.
- **Fields:** PX_LAST, PX_OPEN, PX_HIGH, PX_LOW, PX_VOLUME, EQY_TOTAL_RETURN_INDEX.
- **Universe:** 40–50 tickers (S&P 500, Technology + Financial Services).
- **Output CSV:** `data/raw/prices_raw.csv` (or equivalent name).

## Pull 2 — Static equity fields (one-time)

- **Fields:** CUR_MKT_CAP, BETA_RAW_OVERRIDABLE, DVD_YLD_IND, GICS_SECTOR_NAME per ticker.
- **Output CSV:** `data/raw/static_fields.csv`.

## Pull 3 — Macro (daily)

- **Indices:** VIX Index, USGG10YR Index, SPX Index (e.g. total return).
- **Output CSV:** `data/raw/macro.csv`.

## Delivery

- Copy CSVs to `data/raw/` (gitignored). Ingest via `pipeline/ingest.py` (Phase 1).
