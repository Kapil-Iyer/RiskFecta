/**
 * Known Bloomberg data cutoff (BUILD_PLAN.md / README.md: "Bloomberg history
 * ends around 2026-02-27 and must never be described as live/current beyond
 * that date"). Used only as an immediate fallback for the data-freshness
 * banner before GET /api/market/summary resolves — the live `last_date`
 * value always takes priority once it loads.
 */
export const KNOWN_DATA_CUTOFF = "2026-02-27";
