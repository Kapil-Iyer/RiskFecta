/**
 * Centralized RiskFecta API client. Every fetch call to the FastAPI backend
 * goes through this module — components never call `fetch` directly.
 *
 * Base URL comes from VITE_API_BASE_URL (see .env.example); it defaults to
 * the local Phase 2A dev server so `npm run dev` works out of the box.
 */
import type {
  HealthResponse,
  MarketSummaryResponse,
  PredictionCrossSectionResponse,
  PriceHistoryResponse,
  ReadinessResponse,
  UniverseTicker,
} from "./types";

export const API_BASE_URL: string =
  (import.meta.env.VITE_API_BASE_URL as string | undefined)?.replace(/\/$/, "") ??
  "http://127.0.0.1:8000";

/** Uniform, UI-safe error type. `message` is always safe to render directly
 * — it is either the backend's own `detail` string (Phase 2A never puts
 * credentials or connection details there) or a generic client-side message,
 * never a raw exception/stack trace. */
export class ApiError extends Error {
  readonly status: number | null;

  constructor(message: string, status: number | null = null) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

async function request<T>(path: string, params?: Record<string, string | undefined>): Promise<T> {
  const url = new URL(`${API_BASE_URL}${path}`);
  if (params) {
    for (const [key, value] of Object.entries(params)) {
      if (value !== undefined && value !== "") {
        url.searchParams.set(key, value);
      }
    }
  }

  let response: Response;
  try {
    response = await fetch(url.toString(), { headers: { Accept: "application/json" } });
  } catch {
    // Network-level failure (backend down, DNS, CORS block, offline, etc.) —
    // never surface the raw TypeError to the UI.
    throw new ApiError("Unable to reach the RiskFecta API. Is the backend running?");
  }

  if (!response.ok) {
    let detail = `Request failed (${response.status})`;
    try {
      const body = (await response.json()) as { detail?: string };
      if (body?.detail) detail = body.detail;
    } catch {
      // Non-JSON error body — fall back to the generic message above.
    }
    throw new ApiError(detail, response.status);
  }

  return (await response.json()) as T;
}

export function getHealth(): Promise<HealthResponse> {
  return request<HealthResponse>("/health");
}

export function getReadiness(): Promise<ReadinessResponse> {
  return request<ReadinessResponse>("/health/ready");
}

export function getUniverse(): Promise<UniverseTicker[]> {
  return request<UniverseTicker[]>("/api/universe");
}

export function getPrices(
  ticker: string,
  range?: { start?: string; end?: string },
): Promise<PriceHistoryResponse> {
  return request<PriceHistoryResponse>(`/api/prices/${encodeURIComponent(ticker)}`, {
    start: range?.start,
    end: range?.end,
  });
}

export function getMarketSummary(): Promise<MarketSummaryResponse> {
  return request<MarketSummaryResponse>("/api/market/summary");
}

/** All historical walk-forward formation dates with persisted forecasts
 * (the frozen Phase 4-6 calendar), ascending. */
export function getPredictionDates(): Promise<string[]> {
  return request<string[]>("/api/predictions/dates");
}

/** The 50-ticker forecast cross-section for a formation date. Omit
 * `formationDate` to get the latest available — never a live forecast,
 * just the most recent frozen historical one. */
export function getPredictions(formationDate?: string): Promise<PredictionCrossSectionResponse> {
  return request<PredictionCrossSectionResponse>("/api/predictions", { formation_date: formationDate });
}
