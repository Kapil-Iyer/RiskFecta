import { useEffect, useRef } from "react";
import type { Config, Data, Layout } from "plotly.js-cartesian-dist-min";

/**
 * Thin React wrapper around plotly.js-cartesian-dist-min (line/bar charts
 * only — no map/3D bundle, so no unnecessary/vulnerable transitive deps).
 * There is no official React wrapper for this lighter build, so this
 * component talks to the imperative Plotly API directly: react() to render
 * and efficiently diff-update, plus a ResizeObserver so charts stay
 * responsive in a flex/grid layout.
 *
 * Plotly is loaded via a dynamic import (not a top-level one) purely to keep
 * it out of the initial JS bundle — it's only needed once a ticker is
 * selected, not on first paint.
 *
 * `generationRef` guards against a React StrictMode dev-mode race: the
 * render effect and the unmount-purge effect are both async (they await the
 * dynamic import), and StrictMode's mount→cleanup→mount replay can otherwise
 * let a stale cleanup's `purge()` land *after* the real remount has already
 * drawn the chart, wiping it back to an empty div. Each render effect run
 * bumps the generation counter; a pending purge only actually executes if no
 * newer generation has since rendered into the same node.
 */
interface PlotlyChartProps {
  data: Data[];
  layout?: Partial<Layout>;
  config?: Partial<Config>;
  className?: string;
  ariaLabel?: string;
}

const DEFAULT_CONFIG: Partial<Config> = {
  responsive: true,
  displaylogo: false,
  modeBarButtonsToRemove: ["lasso2d", "select2d"],
};

export default function PlotlyChart({ data, layout, config, className, ariaLabel }: PlotlyChartProps) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const generationRef = useRef(0);

  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;

    const myGeneration = ++generationRef.current;
    let observer: ResizeObserver | undefined;

    import("plotly.js-cartesian-dist-min").then((mod) => {
      if (generationRef.current !== myGeneration) return; // superseded before Plotly finished loading
      const Plotly = mod.default;
      void Plotly.react(el, data, layout ?? {}, { ...DEFAULT_CONFIG, ...config });

      observer = new ResizeObserver(() => {
        if (generationRef.current === myGeneration) void Plotly.Plots.resize(el);
      });
      observer.observe(el);
    });

    return () => {
      observer?.disconnect();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [data, layout, config]);

  useEffect(() => {
    const el = containerRef.current;
    const generationAtMount = generationRef.current;

    return () => {
      import("plotly.js-cartesian-dist-min").then((mod) => {
        // Only purge if this was truly the last render for `el` — not a
        // StrictMode dev-mode replay superseded by a newer generation.
        if (!el || generationRef.current !== generationAtMount) return;
        mod.default.purge(el);
      });
    };
  }, []);

  return <div ref={containerRef} className={className} role="img" aria-label={ariaLabel} />;
}
