export default function MethodologySection() {
  return (
    <section className="card methodology" aria-labelledby="methodology-heading">
      <h2 id="methodology-heading">Methodology &amp; roadmap</h2>
      <p>
        RiskFecta sources institutional-grade historical market and macro data from the Bloomberg
        Terminal (BQL export) for a fixed 50-stock universe (25 Information Technology + 25
        Financials). Data is validated, normalized, and served from a FastAPI backend backed by
        Supabase-hosted PostgreSQL — every number on this page is read directly from that database.
      </p>

      <h3>Planned pipeline (not yet built)</h3>
      <ol className="roadmap-list">
        <li>
          <strong>Feature engineering</strong> <span className="tag tag--planned">planned</span> —
          leakage-safe technical indicators and macro alignment, backward-looking only.
        </li>
        <li>
          <strong>XGBoost + LSTM ensemble</strong> <span className="tag tag--planned">planned</span>{" "}
          — pooled models forecasting 21-trading-session forward returns, evaluated under
          walk-forward (never random) splits.
        </li>
        <li>
          <strong>Covariance &amp; risk estimation</strong>{" "}
          <span className="tag tag--planned">planned</span> — historical-realized-return
          covariance (sample vs. Ledoit-Wolf shrinkage).
        </li>
        <li>
          <strong>Constrained portfolio optimization</strong>{" "}
          <span className="tag tag--planned">planned</span> — long-only mean-variance
          optimization producing an Efficient Frontier.
        </li>
      </ol>

      <p className="methodology__note">
        No forecast, backtest, or portfolio result exists in this project yet. Nothing on this page
        is a fabricated or illustrative substitute for those — see <code>PRD.md</code>,{" "}
        <code>TRD.md</code>, <code>ML_SPEC.md</code>, and <code>BUILD_PLAN.md</code> in the
        repository for the full, frozen specification.
      </p>
    </section>
  );
}
