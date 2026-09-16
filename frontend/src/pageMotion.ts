/** Shared entrance transition for each routed page's root element. Kept
 * near-instant automatically under `MotionConfig reducedMotion="user"` in
 * AppShell — no per-page reduced-motion plumbing needed. */
export const sectionMotion = {
  initial: { opacity: 0, y: 10 },
  animate: { opacity: 1, y: 0 },
};

export const sectionTransition = { duration: 0.3, delay: 0.02 };
