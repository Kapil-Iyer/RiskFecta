import { describe, expect, it } from "vitest";
import COMPANY_NAMES, { companyName } from "./companyNames";
import { TICKER_UNIVERSE } from "./tickerUniverse";

describe("companyNames metadata", () => {
  it("has exactly one company name for every configured ticker", () => {
    for (const ticker of TICKER_UNIVERSE) {
      expect(COMPANY_NAMES[ticker]).toBeTruthy();
    }
    expect(Object.keys(COMPANY_NAMES).sort()).toEqual([...TICKER_UNIVERSE].sort());
  });

  it("has exactly 50 entries — the locked universe size", () => {
    expect(TICKER_UNIVERSE).toHaveLength(50);
    expect(Object.keys(COMPANY_NAMES)).toHaveLength(50);
  });

  it("defines no metadata for tickers outside the configured universe", () => {
    const universeSet = new Set(TICKER_UNIVERSE);
    for (const ticker of Object.keys(COMPANY_NAMES)) {
      expect(universeSet.has(ticker)).toBe(true);
    }
  });

  it("represents MRSH (Marsh & McLennan) correctly and never the stale MMC ticker", () => {
    expect(COMPANY_NAMES.MRSH).toBe("Marsh & McLennan Companies, Inc.");
    expect(COMPANY_NAMES.MMC).toBeUndefined();
    expect(TICKER_UNIVERSE).not.toContain("MMC");
  });

  it("has no blank, whitespace-only, or placeholder company names", () => {
    for (const [ticker, name] of Object.entries(COMPANY_NAMES)) {
      expect(name.trim().length, `${ticker} has a blank name`).toBeGreaterThan(0);
      expect(name).not.toMatch(/^(TODO|TBD|unknown|N\/A)$/i);
    }
  });

  it("companyName() falls back to the ticker itself for anything outside the universe, never blank", () => {
    expect(companyName("aapl")).toBe("Apple Inc.");
    expect(companyName("ZZZNOTAREALTICKER")).toBe("ZZZNOTAREALTICKER");
  });
});
