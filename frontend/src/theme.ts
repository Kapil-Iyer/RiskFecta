import type { Sector } from "./data/tickerUniverse";

/**
 * Restrained, sector-based accent theming (Phase 2C §F): Information
 * Technology gets a cool blue/cyan family, Financials a muted amber/gold
 * family. These are NOT corporate trademark colors — just two accent
 * families layered onto RiskFecta's one dark identity (background,
 * typography, and layout never change per sector).
 *
 * `hex` values are used where CSS custom properties can't reach (Plotly
 * trace colors, which need a real color string, not `var(--x)`).
 */
export interface SectorTheme {
  hex: string;
  cssVar: string;
}

const IT_THEME: SectorTheme = { hex: "#38bdf8", cssVar: "var(--sector-it)" };
const FIN_THEME: SectorTheme = { hex: "#d9a441", cssVar: "var(--sector-fin)" };
const NEUTRAL_THEME: SectorTheme = { hex: "#5b9dff", cssVar: "var(--color-accent)" };

export function themeForSector(sector: Sector | string | null | undefined): SectorTheme {
  if (sector === "Information Technology") return IT_THEME;
  if (sector === "Financials") return FIN_THEME;
  return NEUTRAL_THEME;
}

/** data-sector value used purely for CSS attribute-selector theming. */
export function sectorAttr(sector: Sector | string | null | undefined): string | undefined {
  if (sector === "Information Technology" || sector === "Financials") return sector;
  return undefined;
}
