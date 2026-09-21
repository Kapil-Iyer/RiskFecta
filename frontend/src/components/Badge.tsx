import type { ReactNode } from "react";

interface BadgeProps {
  children: ReactNode;
  tone?: "neutral" | "live" | "planned" | "accent";
  className?: string;
}

/**
 * A small, local badge/pill primitive, styled via the plain-CSS `.badge`
 * rules in index.css rather than shadcn/Tailwind. Predates this app's later
 * shadcn/ui adoption (see components/ui/) and hasn't been migrated — see the
 * Phase 8F-B0 design-system audit for the consolidation plan.
 */
export default function Badge({ children, tone = "neutral", className }: BadgeProps) {
  return <span className={`badge badge--${tone}${className ? ` ${className}` : ""}`}>{children}</span>;
}
