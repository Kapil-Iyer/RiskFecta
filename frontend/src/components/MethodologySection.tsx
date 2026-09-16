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

      <h3>Research pipeline status</h3>
      <ol className="roadmap-list">
        <li>
          <strong>Feature engineering</strong> <span className="tag tag--research">research complete</span> —
          leakage-safe technical indicators and macro alignment, backward-looking only (Phase 3).
          Feeds every model below; has no dedicated page of its own.
        </li>
        <li>
          <strong>XGBoost + LSTM ensemble</strong> <span className="tag tag--live">live</span> —
          pooled models forecasting 21-trading-session forward returns, evaluated under
          walk-forward (never random) splits. See Forecast Rankings and Model Comparison.
        </li>
        <li>
          <strong>Covariance &amp; risk estimation</strong>{" "}
          <span className="tag tag--research">research complete</span> — historical-realized-return
          covariance (sample vs. Ledoit-Wolf shrinkage), complete in the research pipeline
          (Phase 7). The Risk Analytics dashboard page is not yet built.
        </li>
        <li>
          <strong>Constrained portfolio optimization</strong>{" "}
          <span className="tag tag--research">research complete</span> — long-only mean-variance
          optimization producing an Efficient Frontier, complete in the research pipeline
          (Phase 7). The Portfolio Construction and Efficient Frontier dashboard pages are not
          yet built.
        </li>
      </ol>

      <p className="methodology__note">
        Historical walk-forward forecasts (Phase 4–6) exist and are shown on the Forecast Rankings
        page. Portfolio and backtest results (Phase 7) exist in the underlying research pipeline
        but are not yet surfaced as dashboard pages — see Build status on the Overview page.
        Nothing on this page is a fabricated or illustrative substitute for a real result — see{" "}
        <code>PRD.md</code>, <code>TRD.md</code>, <code>ML_SPEC.md</code>, and{" "}
        <code>BUILD_PLAN.md</code> in the repository for the full, frozen specification.
      </p>
    </section>
  );
}
