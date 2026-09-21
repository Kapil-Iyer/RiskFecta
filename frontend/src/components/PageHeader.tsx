import type { ReactNode } from "react";

interface PageHeaderProps {
  title: string;
  description?: ReactNode;
}

/** Shared page-title block — title + optional description. Used by six
 * routed pages (Forecasts, Models, Portfolio, Frontier, Risk, Backtest);
 * Overview has its own hero, Universe has its own two-pane layout, and
 * Methodology's richer header (extra chip row) was intentionally left as
 * its own markup rather than forced through this primitive. Extracted from
 * an identical hand-rolled `<div><h1>…</h1><p>…</p></div>` block that those
 * six pages previously duplicated. */
export default function PageHeader({ title, description }: PageHeaderProps) {
  return (
    <div className="flex flex-col gap-1">
      <h1 className="text-xl font-semibold text-foreground">{title}</h1>
      {description && <p className="max-w-3xl text-sm text-muted-foreground">{description}</p>}
    </div>
  );
}
