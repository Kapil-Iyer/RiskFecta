import type { UniverseSceneNode } from "./universeScene";
import { themeForSector } from "../../theme";

interface UniverseFallbackGridProps {
  nodes: UniverseSceneNode[];
}

/**
 * Accessible, non-WebGL universe view: real tickers, real sector coloring,
 * no fabricated data. Used when WebGL is unavailable, and always present in
 * the DOM as a screen-reader-visible summary alongside the 3D view.
 */
export default function UniverseFallbackGrid({ nodes }: UniverseFallbackGridProps) {
  return (
    <div
      className="grid grid-cols-5 gap-1.5 sm:grid-cols-8 md:grid-cols-10"
      role="list"
      aria-label={`${nodes.length}-stock RiskFecta universe`}
    >
      {nodes.map((n) => {
        const theme = themeForSector(n.sector);
        return (
          <div
            key={n.ticker}
            role="listitem"
            title={n.sector}
            className="flex items-center gap-1.5 rounded-md border border-border bg-secondary/40 px-1.5 py-1 font-mono text-[10px] text-foreground"
          >
            <span className="h-1.5 w-1.5 shrink-0 rounded-full" style={{ backgroundColor: theme.hex }} aria-hidden="true" />
            {n.ticker}
          </div>
        );
      })}
    </div>
  );
}
