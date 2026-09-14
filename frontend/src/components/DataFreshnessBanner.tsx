import { KNOWN_DATA_CUTOFF } from "../constants";

interface DataFreshnessBannerProps {
  /** Live `last_date` from GET /api/market/summary, once loaded. Falls back
   * to the documented Bloomberg cutoff (constants.ts) until then — never
   * presented as "live" or current-as-of-today. */
  lastDate?: string | null;
}

export default function DataFreshnessBanner({ lastDate }: DataFreshnessBannerProps) {
  const asOf = lastDate ?? KNOWN_DATA_CUTOFF;
  return (
    <div className="freshness-banner" role="note">
      <strong>Historical data through {asOf}</strong>
      <span aria-hidden="true"> · </span>
      <span>Model forecasts in development</span>
    </div>
  );
}
