import Badge from "./Badge";

interface StatusItem {
  label: string;
  state: "live" | "planned";
  detail: string;
}

const STATUS_ITEMS: StatusItem[] = [
  {
    label: "Historical market data",
    state: "live",
    detail: "Bloomberg-sourced OHLCV + total return index, 50-stock universe, live in PostgreSQL.",
  },
  {
    label: "Model forecasts",
    state: "planned",
    detail: "Pooled XGBoost + LSTM ensemble — not yet trained. No predictions exist.",
  },
  {
    label: "Portfolio optimization",
    state: "planned",
    detail: "Constrained mean-variance optimization + Efficient Frontier — not yet built.",
  },
];

export default function BuildStatusSection() {
  return (
    <section className="card build-status" aria-labelledby="build-status-heading">
      <h2 id="build-status-heading">Build status</h2>
      <ul className="build-status__list">
        {STATUS_ITEMS.map((item) => (
          <li key={item.label} className={`build-status__item build-status__item--${item.state}`}>
            <Badge tone={item.state}>{item.state === "live" ? "Live" : "Planned"}</Badge>
            <div>
              <p className="build-status__label">{item.label}</p>
              <p className="build-status__detail">{item.detail}</p>
            </div>
          </li>
        ))}
      </ul>
    </section>
  );
}
