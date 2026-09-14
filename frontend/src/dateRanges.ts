/** Quick-range + date-display helpers for TickerDetail. Every range here is
 * computed client-side from a ticker's own already-known latest date — no
 * new endpoint, no fabricated boundary. "ALL" simply clears the filter and
 * reuses the existing unfiltered fetch. */
export type QuickRangeKey = "1M" | "6M" | "1Y" | "3Y" | "ALL";

export const QUICK_RANGES: QuickRangeKey[] = ["1M", "6M", "1Y", "3Y", "ALL"];

export function computeQuickRange(latestDate: string, key: QuickRangeKey): { start: string; end: string } | null {
  if (key === "ALL") return null;
  const end = latestDate;
  const d = new Date(`${latestDate}T00:00:00Z`);
  switch (key) {
    case "1M":
      d.setUTCMonth(d.getUTCMonth() - 1);
      break;
    case "6M":
      d.setUTCMonth(d.getUTCMonth() - 6);
      break;
    case "1Y":
      d.setUTCFullYear(d.getUTCFullYear() - 1);
      break;
    case "3Y":
      d.setUTCFullYear(d.getUTCFullYear() - 3);
      break;
  }
  return { start: d.toISOString().slice(0, 10), end };
}

/** "2026-02-27" -> "Feb 27, 2026" */
export function formatLongDate(iso: string): string {
  return new Date(`${iso}T00:00:00Z`).toLocaleDateString("en-US", {
    month: "short",
    day: "numeric",
    year: "numeric",
    timeZone: "UTC",
  });
}

/** "2021-03-01" -> "Mar 2021" */
export function formatMonthYear(iso: string): string {
  return new Date(`${iso}T00:00:00Z`).toLocaleDateString("en-US", {
    month: "short",
    year: "numeric",
    timeZone: "UTC",
  });
}
