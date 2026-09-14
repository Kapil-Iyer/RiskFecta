export default function Header() {
  return (
    <header className="app-header">
      <div className="app-header__brand">
        <span className="app-header__mark" aria-hidden="true">
          RF
        </span>
        <div>
          <h1>RiskFecta</h1>
          <p className="app-header__tagline">Quantitative Portfolio Intelligence Platform</p>
        </div>
      </div>
      <p className="app-header__disclaimer">
        Research &amp; analytics project — not a trading platform, brokerage, or source of investment advice.
      </p>
    </header>
  );
}
