/**
 * Mirrors config.TICKER_UNIVERSE (the repo's Python source of truth for the
 * locked 50-stock universe) so frontend-only metadata (company names —
 * companyNames.ts) can be validated for completeness without a backend
 * round trip. This is NOT an independent source of truth: the authoritative
 * list lives in config.py, keyed to the Bloomberg data/raw export.
 *
 * Keep in sync manually. If config.TICKER_UNIVERSE ever changes, update this
 * array (and companyNames.ts) in the same change — companyNames.test.ts
 * fails loudly if the two ever drift apart.
 */
export const TICKER_UNIVERSE: readonly string[] = [
  // Information Technology (25) — order matches config.py
  "AAPL", "MSFT", "NVDA", "AVGO", "ORCL", "CRM", "ADBE", "AMD", "QCOM", "TXN",
  "INTC", "MU", "AMAT", "KLAC", "SNPS", "CDNS", "PANW", "NOW", "APH", "MSI",
  "ADI", "MRVL", "IBM", "HPQ", "GLW",
  // Financials (25) — order matches config.py; MRSH as Bloomberg-exported, not MMC
  "JPM", "BAC", "WFC", "GS", "MS", "C", "BLK", "SCHW", "AXP", "USB",
  "PNC", "TFC", "COF", "MRSH", "ICE", "CME", "SPGI", "MCO", "MSCI", "CB",
  "PGR", "MET", "AIG", "TRV", "AJG",
];

export type Sector = "Information Technology" | "Financials";

const IT_TICKERS = new Set(TICKER_UNIVERSE.slice(0, 25));

/** Independent of (and a cross-check for) the sector string the API may
 * return from the static snapshot — used only for client-side accent
 * theming, never presented as if it came from the backend. */
export function sectorForTicker(ticker: string): Sector | null {
  const t = ticker.toUpperCase();
  if (!TICKER_UNIVERSE.includes(t)) return null;
  return IT_TICKERS.has(t) ? "Information Technology" : "Financials";
}
