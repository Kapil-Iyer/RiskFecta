import { useCallback, useState } from "react";
import { getReadiness } from "../api/client";
import { useAsync } from "../hooks/useAsync";

/**
 * Surfaces backend-unavailable / database-disconnected states using
 * GET /health/ready (liveness alone — GET /health — can't tell readers
 * whether Postgres is actually reachable). Renders nothing when healthy.
 */
export default function BackendStatusBanner() {
  const [attempt, setAttempt] = useState(0);
  const state = useAsync(getReadiness, [attempt]);
  const retry = useCallback(() => setAttempt((n) => n + 1), []);

  if (state.status === "loading") return null;
  if (state.status === "success" && state.data.status === "ok") return null;

  const message =
    state.status === "error"
      ? "Backend unavailable — the RiskFecta API could not be reached."
      : "Database unavailable — the API is up but cannot reach PostgreSQL right now.";

  return (
    <div className="status-banner status-banner--error" role="alert">
      <span>{message}</span>
      <button type="button" className="retry-button" onClick={retry}>
        Retry
      </button>
    </div>
  );
}
