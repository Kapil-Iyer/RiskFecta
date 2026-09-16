export interface NavItem {
  path: string;
  label: string;
}

/** Single source of truth for the app's nav links and route paths — see
 * AppRoutes.tsx, which renders a route for each of these. */
export const NAV_ITEMS: NavItem[] = [
  { path: "/", label: "Overview" },
  { path: "/universe", label: "Universe" },
  { path: "/forecasts", label: "Forecast Rankings" },
  { path: "/models", label: "Model Comparison" },
  { path: "/portfolio", label: "Portfolio Construction" },
  { path: "/frontier", label: "Efficient Frontier" },
  { path: "/risk", label: "Risk Analytics" },
  { path: "/backtest", label: "Historical Evidence" },
  { path: "/methodology", label: "Methodology" },
];
