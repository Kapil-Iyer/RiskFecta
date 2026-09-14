import { describe, expect, it } from "vitest";
import { computeQuickRange, formatLongDate, formatMonthYear } from "./dateRanges";

describe("computeQuickRange", () => {
  it("ALL clears the filter (returns null)", () => {
    expect(computeQuickRange("2026-02-27", "ALL")).toBeNull();
  });

  it("1M is exactly one calendar month before the latest date", () => {
    expect(computeQuickRange("2026-02-27", "1M")).toEqual({ start: "2026-01-27", end: "2026-02-27" });
  });

  it("1Y is exactly one calendar year before the latest date", () => {
    expect(computeQuickRange("2026-02-27", "1Y")).toEqual({ start: "2025-02-27", end: "2026-02-27" });
  });

  it("3Y never precedes the ticker's own latest date as the end bound", () => {
    const { end } = computeQuickRange("2026-02-27", "3Y")!;
    expect(end).toBe("2026-02-27");
  });
});

describe("date display formatting", () => {
  it("formats a long date", () => {
    expect(formatLongDate("2026-02-27")).toBe("Feb 27, 2026");
  });

  it("formats a month/year", () => {
    expect(formatMonthYear("2021-03-01")).toBe("Mar 2021");
  });
});
