import { afterEach, describe, expect, it, vi } from "vitest";
import { ApiError, getMarketSummary, getPrices, getUniverse } from "./client";

function jsonResponse(body: unknown, init?: ResponseInit): Response {
  return new Response(JSON.stringify(body), {
    status: 200,
    headers: { "Content-Type": "application/json" },
    ...init,
  });
}

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("api client", () => {
  it("returns parsed JSON on a successful request", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(
        jsonResponse({ ticker_count: 50, price_row_count: 62800, first_date: "2021-03-01", last_date: "2026-02-27" }),
      ),
    );

    const summary = await getMarketSummary();
    expect(summary.ticker_count).toBe(50);
    expect(summary.price_row_count).toBe(62800);
  });

  it("surfaces the backend's `detail` message on a non-2xx response", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(jsonResponse({ detail: "Unknown ticker: ZZZ" }, { status: 404 })),
    );

    await expect(getPrices("ZZZ")).rejects.toMatchObject({
      message: "Unknown ticker: ZZZ",
      status: 404,
    });
  });

  it("never leaks a raw network exception — only a generic ApiError", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockRejectedValue(new TypeError("Failed to fetch: connect ECONNREFUSED 127.0.0.1:8000")),
    );

    const error = await getUniverse().catch((e: unknown) => e);
    expect(error).toBeInstanceOf(ApiError);
    expect((error as ApiError).message).not.toMatch(/ECONNREFUSED/);
    expect((error as ApiError).message).toMatch(/Unable to reach the RiskFecta API/);
  });

  it("passes start/end as query params to /api/prices/{ticker}", async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      jsonResponse({ ticker: "AAPL", start: "2025-01-01", end: "2025-01-31", count: 0, prices: [] }),
    );
    vi.stubGlobal("fetch", fetchMock);

    await getPrices("AAPL", { start: "2025-01-01", end: "2025-01-31" });

    const requestedUrl = new URL(fetchMock.mock.calls[0][0] as string);
    expect(requestedUrl.pathname).toBe("/api/prices/AAPL");
    expect(requestedUrl.searchParams.get("start")).toBe("2025-01-01");
    expect(requestedUrl.searchParams.get("end")).toBe("2025-01-31");
  });
});
