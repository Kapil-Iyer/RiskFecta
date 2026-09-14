import type { ReactNode } from "react";

interface BadgeProps {
  children: ReactNode;
  tone?: "neutral" | "live" | "planned" | "accent";
  className?: string;
}

/**
 * A small, local badge/pill primitive. Phase 2C considered shadcn/ui for
 * this (see README/report), but shadcn requires adopting Tailwind CSS —
 * a disproportionate footprint change for a handful of primitives on an app
 * that otherwise uses plain CSS custom properties. This covers the same
 * need (status pill, sector tag) without a new styling paradigm.
 */
export default function Badge({ children, tone = "neutral", className }: BadgeProps) {
  return <span className={`badge badge--${tone}${className ? ` ${className}` : ""}`}>{children}</span>;
}
