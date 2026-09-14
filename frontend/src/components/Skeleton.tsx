interface SkeletonProps {
  width?: string | number;
  height?: string | number;
  className?: string;
}

/** A local loading-placeholder primitive (see Badge.tsx for why this isn't
 * shadcn/ui's Skeleton). Respects prefers-reduced-motion via the `.skeleton`
 * CSS rule (pulse animation disabled, static tone shown instead). */
export default function Skeleton({ width = "100%", height = "1em", className }: SkeletonProps) {
  return (
    <span
      className={`skeleton${className ? ` ${className}` : ""}`}
      style={{ width, height }}
      aria-hidden="true"
    />
  );
}
