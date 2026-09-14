/**
 * Human-readable company names for the locked 50-stock universe
 * (config.TICKER_UNIVERSE — mirrored in tickerUniverse.ts).
 *
 * DATA-INTEGRITY NOTE (PRD.md / ML_SPEC.md §2, §10):
 * The Bloomberg static-fields export (data/raw/static_fields.csv) carries
 * only CUR_MKT_CAP, BETA_RAW_OVERRIDABLE, DIVIDEND_INDICATED_YIELD, and
 * GICS_SECTOR_NAME — it does NOT include a company-name field. These names
 * are therefore explicit, hand-maintained APPLICATION metadata (official
 * legal/display names, from public company knowledge), not a Bloomberg
 * field, not a predictive feature, and not something historically attached
 * to a `prices_raw` observation. They are display-only and change nothing
 * about how any table is queried or stored.
 *
 * Single source of truth for this metadata: this file. Do not duplicate
 * ticker → name mappings elsewhere in the app.
 *
 * KNOWN TICKER-SYMBOL AMBIGUITY (flagged, not guessed):
 * MRSH — the Bloomberg export uses ticker "MRSH" for Marsh & McLennan
 * Companies, Inc. (see config.py comment: "MRSH as exported — not MMC").
 * MMC is that company's standard market ticker elsewhere; this project's
 * data pipeline is intentionally keyed to "MRSH" throughout (Phase 1A/1B
 * audits), so the entry below is filed under MRSH to match `prices_raw` and
 * `config.TICKER_UNIVERSE` exactly. The company identity itself (Marsh &
 * McLennan Companies, Inc.) is not in doubt — only the ticker spelling.
 */
import { TICKER_UNIVERSE } from "./tickerUniverse";

const COMPANY_NAMES: Readonly<Record<string, string>> = {
  // Information Technology
  AAPL: "Apple Inc.",
  MSFT: "Microsoft Corporation",
  NVDA: "NVIDIA Corporation",
  AVGO: "Broadcom Inc.",
  ORCL: "Oracle Corporation",
  CRM: "Salesforce, Inc.",
  ADBE: "Adobe Inc.",
  AMD: "Advanced Micro Devices, Inc.",
  QCOM: "QUALCOMM Incorporated",
  TXN: "Texas Instruments Incorporated",
  INTC: "Intel Corporation",
  MU: "Micron Technology, Inc.",
  AMAT: "Applied Materials, Inc.",
  KLAC: "KLA Corporation",
  SNPS: "Synopsys, Inc.",
  CDNS: "Cadence Design Systems, Inc.",
  PANW: "Palo Alto Networks, Inc.",
  NOW: "ServiceNow, Inc.",
  APH: "Amphenol Corporation",
  MSI: "Motorola Solutions, Inc.",
  ADI: "Analog Devices, Inc.",
  MRVL: "Marvell Technology, Inc.",
  IBM: "International Business Machines Corporation",
  HPQ: "HP Inc.",
  GLW: "Corning Incorporated",

  // Financials
  JPM: "JPMorgan Chase & Co.",
  BAC: "Bank of America Corporation",
  WFC: "Wells Fargo & Company",
  GS: "The Goldman Sachs Group, Inc.",
  MS: "Morgan Stanley",
  C: "Citigroup Inc.",
  BLK: "BlackRock, Inc.",
  SCHW: "The Charles Schwab Corporation",
  AXP: "American Express Company",
  USB: "U.S. Bancorp",
  PNC: "The PNC Financial Services Group, Inc.",
  TFC: "Truist Financial Corporation",
  COF: "Capital One Financial Corporation",
  MRSH: "Marsh & McLennan Companies, Inc.", // ticker-symbol note above — not MMC
  ICE: "Intercontinental Exchange, Inc.",
  CME: "CME Group Inc.",
  SPGI: "S&P Global Inc.",
  MCO: "Moody's Corporation",
  MSCI: "MSCI Inc.",
  CB: "Chubb Limited",
  PGR: "The Progressive Corporation",
  MET: "MetLife, Inc.",
  AIG: "American International Group, Inc.",
  TRV: "The Travelers Companies, Inc.",
  AJG: "Arthur J. Gallagher & Co.",
};

// Fail loudly at build/import time (not silently at render time) if this
// map and the authoritative universe mirror ever drift — belt-and-suspenders
// alongside companyNames.test.ts.
const missing = TICKER_UNIVERSE.filter((t) => !COMPANY_NAMES[t]);
if (missing.length > 0) {
  throw new Error(`companyNames.ts is missing entries for: ${missing.join(", ")}`);
}

/** Returns the display name for a configured ticker, or the ticker itself
 * (never blank, never throws) if it falls outside the locked universe. */
export function companyName(ticker: string): string {
  return COMPANY_NAMES[ticker.toUpperCase()] ?? ticker.toUpperCase();
}

export default COMPANY_NAMES;
