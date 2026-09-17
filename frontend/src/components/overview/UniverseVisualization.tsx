import { useEffect, useRef, useState } from "react";
import type { UniverseTicker } from "../../api/types";
import { sectorForTicker } from "../../data/tickerUniverse";
import type { HoverInfo, SceneHandle, UniverseSceneNode } from "./universeScene";
import UniverseFallbackGrid from "./UniverseFallbackGrid";

interface UniverseVisualizationProps {
  /** Real rows from GET /api/universe — not synthetic/example data. */
  universe: UniverseTicker[];
}

function supportsWebGL(): boolean {
  try {
    const canvas = document.createElement("canvas");
    return !!(canvas.getContext("webgl") || canvas.getContext("experimental-webgl"));
  } catch {
    return false;
  }
}

function prefersReducedMotion(): boolean {
  try {
    return window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  } catch {
    return true; // safest default when detection itself is unavailable
  }
}

/** Resolves every real ticker to a definite sector (backend `sector` first,
 * falling back to the frontend's own config.py-mirrored table — same
 * pattern as UniverseBrowser) so the visualization always reflects the true
 * 25 Information Technology / 25 Financials split regardless of whether the
 * backend's static-snapshot sector metadata happens to be available. */
function resolveNodes(universe: UniverseTicker[]): UniverseSceneNode[] {
  return universe
    .map((u) => ({ ticker: u.ticker, sector: u.sector ?? sectorForTicker(u.ticker) }))
    .filter((n): n is UniverseSceneNode => n.sector === "Information Technology" || n.sector === "Financials");
}

/**
 * Overview hero visualization: a custom Three.js scene when WebGL is
 * available and motion is welcome, otherwise a static, equally accurate
 * fallback grid — never decorative-only, never implying a relationship
 * (correlation/covariance/edges) this app never computed.
 */
export default function UniverseVisualization({ universe }: UniverseVisualizationProps) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const [hover, setHover] = useState<HoverInfo | null>(null);
  const [webglOk] = useState(supportsWebGL);
  const [reducedMotion] = useState(prefersReducedMotion);

  const nodes = resolveNodes(universe);
  const itCount = nodes.filter((n) => n.sector === "Information Technology").length;
  const finCount = nodes.filter((n) => n.sector === "Financials").length;

  useEffect(() => {
    if (!webglOk || nodes.length === 0) return;
    const el = containerRef.current;
    if (!el) return;

    let cancelled = false;
    let handle: SceneHandle | undefined;

    import("./universeScene").then(({ createUniverseScene }) => {
      if (cancelled || !el.isConnected) return;
      handle = createUniverseScene(el, nodes, { reducedMotion, onHover: setHover });
    });

    return () => {
      cancelled = true;
      handle?.dispose();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [webglOk, reducedMotion, universe]);

  const summary = `${nodes.length}-stock RiskFecta universe: ${itCount} Information Technology, ${finCount} Financials.`;

  if (!webglOk || nodes.length === 0) {
    return (
      <div className="flex flex-col gap-2">
        <UniverseFallbackGrid nodes={nodes} />
        <p className="sr-only">{summary}</p>
      </div>
    );
  }

  return (
    <div className="relative h-[300px] w-full overflow-hidden rounded-lg border border-border bg-secondary/20 sm:h-[360px]">
      <div ref={containerRef} className="h-full w-full" role="img" aria-label={summary} />
      {hover && (
        <div
          className="pointer-events-none absolute rounded-md border border-border bg-card px-2 py-1 font-mono text-xs text-foreground shadow-sm"
          style={{ left: hover.x + 12, top: hover.y + 12 }}
        >
          {hover.ticker}
        </div>
      )}
      <p className="sr-only">{summary}</p>
    </div>
  );
}
