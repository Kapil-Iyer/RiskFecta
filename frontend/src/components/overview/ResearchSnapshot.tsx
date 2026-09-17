import type { UniverseTicker } from "../../api/types";
import { sectorForTicker } from "../../data/tickerUniverse";

interface ResearchSnapshotProps {
  universe: UniverseTicker[];
  foldCount: number | null;
  predictionCount: number | null;
  targetHorizonSessions: number | null;
  dataThrough: string | null | undefined;
}

interface StatItem {
  label: string;
  value: string;
}

/** Compact, defensible facts only — every value here is either a real API
 * count or the fixed 21-session/50-stock methodology constant, never an ML
 * metric dressed up as a headline number. */
export default function ResearchSnapshot({
  universe,
  foldCount,
  predictionCount,
  targetHorizonSessions,
  dataThrough,
}: ResearchSnapshotProps) {
  const itCount = universe.filter((u) => (u.sector ?? sectorForTicker(u.ticker)) === "Information Technology").length;
  const finCount = universe.filter((u) => (u.sector ?? sectorForTicker(u.ticker)) === "Financials").length;

  const stats: StatItem[] = [
    { label: "Equities", value: universe.length > 0 ? `${universe.length}` : "…" },
    { label: "Sectors", value: itCount > 0 || finCount > 0 ? `${itCount} IT / ${finCount} Financials` : "…" },
    { label: "Walk-forward formations", value: foldCount !== null ? `${foldCount}` : "…" },
    { label: "OOS forecasts scored", value: predictionCount !== null ? predictionCount.toLocaleString() : "…" },
    { label: "Forecast horizon", value: targetHorizonSessions !== null ? `${targetHorizonSessions} sessions` : "…" },
    { label: "Bloomberg data through", value: dataThrough ?? "…" },
  ];

  return (
    <dl className="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-6">
      {stats.map((s) => (
        <div key={s.label} className="rounded-md border border-border bg-secondary/30 px-3 py-2.5">
          <dt className="text-[11px] uppercase tracking-wide text-muted-foreground">{s.label}</dt>
          <dd className="mt-0.5 font-mono text-sm font-semibold text-foreground">{s.value}</dd>
        </div>
      ))}
    </dl>
  );
}
