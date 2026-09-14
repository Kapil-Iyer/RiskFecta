interface LogoProps {
  size?: number;
  className?: string;
}

/**
 * RiskFecta's mark: an abstract efficient-frontier curve — a naive starting
 * point (muted dot) rising to an optimal point on the frontier (accent dot).
 * Original, simple, and conceptually tied to the product (PRD.md §6 —
 * Efficient Frontier) rather than a literal monogram or a generic template
 * icon. Mirrors public/favicon.svg; kept as inline SVG here so it can share
 * the app's color tokens and stay crisp at any size.
 */
export default function Logo({ size = 36, className }: LogoProps) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 40 40"
      className={className}
      role="img"
      aria-label="RiskFecta logomark"
    >
      <rect x="0.75" y="0.75" width="38.5" height="38.5" rx="9" fill="var(--color-logo-bg, #0a0f1c)" stroke="var(--color-border)" strokeWidth="1.5" />
      <path d="M10 29 C 13 13 24 9 30 11" fill="none" stroke="var(--color-accent)" strokeWidth="2.6" strokeLinecap="round" />
      <circle cx="10" cy="29" r="2.1" fill="var(--color-text-muted)" />
      <circle cx="30" cy="11" r="2.8" fill="var(--color-accent)" stroke="var(--color-logo-bg, #0a0f1c)" strokeWidth="1.4" />
    </svg>
  );
}
