import { motion } from "framer-motion";
import { Link } from "react-router-dom";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "../components/ui/card";
import { KNOWN_DATA_CUTOFF } from "../constants";
import { sectionMotion, sectionTransition } from "../pageMotion";

function Chip({ children }: { children: string }) {
  return (
    <span className="rounded-full border border-border bg-secondary/40 px-3 py-1 text-xs text-muted-foreground">
      {children}
    </span>
  );
}

interface PipelineStep {
  title: string;
  detail: string;
  frozen?: string;
  links?: { to: string; label: string }[];
}

const PIPELINE: PipelineStep[] = [
  {
    title: "Data",
    detail:
      "Historical Bloomberg exports for a fixed 50-stock universe (25 Information Technology, 25 Financials): OHLC, volume, and Total Return Index per stock, plus SPX, VIX, and the US 10-year yield as macro context.",
    frozen: `Session validity is gated on PX_LAST (close) — never on Total Return Index. Normalized history: 2021-03-01 through ${KNOWN_DATA_CUTOFF}.`,
    links: [{ to: "/universe", label: "Universe" }],
  },
  {
    title: "Features",
    detail:
      "Causal, backward-looking-only technical indicators and macro alignment. No feature ever uses information dated after the row it describes.",
    frozen: "Feeds every model below — no dedicated dashboard page of its own.",
  },
  {
    title: "Target",
    detail: "21-trading-session forward total return, computed from Bloomberg Total Return Index (TRI), not price.",
    frozen: "target_t = TRI(t+21) / TRI(t) − 1 — 21 valid trading sessions, never 21 calendar days.",
  },
  {
    title: "Forecasts",
    detail:
      "Two pooled models — one XGBoost, one LSTM — trained across the full 50-stock universe and evaluated walk-forward (never a random split).",
    links: [
      { to: "/forecasts", label: "Forecast Rankings" },
      { to: "/models", label: "Model Comparison" },
    ],
  },
  {
    title: "Ensemble",
    detail: "A fixed 50/50 arithmetic mean of the two models' predictions.",
    frozen: "ensemble_pred = 0.5 · XGB_pred + 0.5 · LSTM_pred — frozen before observing walk-forward results, never reweighted afterward.",
    links: [{ to: "/models", label: "Model Comparison" }],
  },
  {
    title: "Covariance",
    detail: "252 sessions of historical realized returns as of the formation date, estimated two ways: Sample and Ledoit-Wolf shrinkage.",
    frozen: "Requires 253 prior TRI levels. Session-frequency Σ is scaled to the 21-session horizon: Σ_21 = 21 · Σ_session.",
  },
  {
    title: "Optimization",
    detail: "Constrained portfolio construction: long-only, fully invested, 10% maximum single-name weight.",
    frozen: "Five strategies: Sample Min-Vol, Sample Max-Sharpe, Ledoit-Wolf Min-Vol, Ledoit-Wolf Max-Sharpe, Equal Weight.",
    links: [
      { to: "/portfolio", label: "Portfolio Construction" },
      { to: "/frontier", label: "Efficient Frontier" },
    ],
  },
  {
    title: "Evaluation",
    detail: "46 sequential, non-overlapping 21-session periods, evaluated against the frozen formation-time weights and the SPXT benchmark.",
    frozen: "Historical walk-forward evidence only — never annualized, never a sealed-holdout result.",
    links: [{ to: "/backtest", label: "Historical Evidence" }],
  },
  {
    title: "Sealed March 2026",
    detail: "A single, untouched post-freeze holdout reserved for Phase 9 — the actual out-of-sample generalization test.",
    frozen: "SEALED — NOT YET EVALUATED.",
  },
];

const LIMITATIONS: string[] = [
  "Historical walk-forward evidence is not proof of future performance.",
  "Standalone return-forecast performance was weak/mixed — the Historical Mean baseline beat both ML models on MAE, RMSE, and directional accuracy (see Model Comparison).",
  "The universe contains only 50 stocks.",
  "The universe is restricted to two sectors: Information Technology and Financials.",
  "Transaction costs and slippage are not modeled anywhere in the pipeline.",
  "The Max-Sharpe portfolios exhibited substantially higher turnover than Min-Vol or Equal Weight.",
  "The hard 10% single-name cap frequently bound for optimized portfolios, materially shaping their composition.",
  "Static snapshot fields (market cap, beta, dividend yield) were excluded from predictive features — no point-in-time historical version of these fields was available, and attaching a current-day snapshot to historical rows would be look-ahead leakage.",
  "Covariance is historical realized covariance, not a predicted or forward-looking covariance.",
  "Portfolios are long-only and fully invested — no leverage, no shorting.",
  "Historical walk-forward results and the sealed March 2026 case study are different evidence categories and should not be conflated.",
  "The sealed March holdout, once evaluated, is a single post-freeze case study — not by itself sufficient to establish broad generalization.",
];

const DOES_NOT_CLAIM: string[] = [
  "Guaranteed returns",
  "Production trading readiness",
  "Live or current-day market coverage",
  "Broad market generalization beyond this 50-stock, two-sector universe",
  "Consistently accurate stock-return prediction",
  "Investment advice",
];

const FROZEN_ITEMS: string[] = [
  "21-session forward TRI target definition",
  "XGBoost and LSTM feature sets and architectures",
  "Ensemble weights (fixed 50/50)",
  "Covariance methodology (252-session window, Sample/Ledoit-Wolf, 21-session scaling)",
  "Optimization constraints (long-only, fully invested, 10% cap) and the five strategies",
  "SPXT as the evaluation benchmark",
  "The walk-forward evaluation protocol",
];

export default function MethodologyPage() {
  return (
    <motion.div {...sectionMotion} transition={sectionTransition} className="app-page">
      {/* A. Hero / research protocol */}
      <div className="flex flex-col gap-2">
        <h1 className="text-xl font-semibold text-foreground">Methodology</h1>
        <p className="max-w-3xl text-sm text-muted-foreground">
          RiskFecta is a causal, walk-forward research pipeline: every step below uses only information available
          at the time, methodology was frozen before the sealed March 2026 holdout was ever touched, and every
          number shown elsewhere in this app traces back to one of the frozen steps described here.
        </p>
        <div className="flex flex-wrap gap-2 pt-1">
          <Chip>Historical Bloomberg data</Chip>
          <Chip>Walk-forward OOS</Chip>
          <Chip>21-session horizon</Chip>
          <Chip>50-stock universe</Chip>
          <Chip>Sealed holdout pending</Chip>
        </div>
      </div>

      {/* B. Research pipeline */}
      <Card>
        <CardHeader>
          <CardTitle>Research pipeline</CardTitle>
          <CardDescription>Each step is causal — it only ever sees information available as of that step's own date.</CardDescription>
        </CardHeader>
        <CardContent className="flex flex-col">
          {PIPELINE.map((step, i) => (
            <div key={step.title} className="relative flex gap-4 pb-6 last:pb-0">
              <div className="flex flex-col items-center">
                <span className="flex h-7 w-7 shrink-0 items-center justify-center rounded-full border border-border bg-secondary/50 font-mono text-xs text-foreground">
                  {i + 1}
                </span>
                {i < PIPELINE.length - 1 && <span aria-hidden="true" className="mt-1 w-px flex-1 bg-border" />}
              </div>
              <div className="flex flex-col gap-1 pt-0.5">
                <h3 className="text-sm font-semibold text-foreground">{step.title}</h3>
                <p className="text-sm text-muted-foreground">{step.detail}</p>
                {step.frozen && <p className="font-mono text-xs text-muted-foreground/80">{step.frozen}</p>}
                {step.links && (
                  <div className="flex gap-3 pt-0.5">
                    {step.links.map((l) => (
                      <Link key={l.to} to={l.to} className="text-xs underline underline-offset-2 hover:text-foreground">
                        {l.label}
                      </Link>
                    ))}
                  </div>
                )}
              </div>
            </div>
          ))}
        </CardContent>
      </Card>

      {/* C. Temporal integrity */}
      <Card>
        <CardHeader>
          <CardTitle>Temporal integrity — no future leakage</CardTitle>
          <CardDescription>
            No random train/test split anywhere in this pipeline. Every model is trained on strictly past
            information and evaluated as it walks forward through time.
          </CardDescription>
        </CardHeader>
        <CardContent className="flex flex-col gap-3 text-sm text-muted-foreground">
          <p>
            At each formation date T, a model is fit using only sessions at or before T. Its forecast for T is
            evaluated only once the 21-session target date T+21 has actually passed. Portfolio construction at T
            never has access to the realized T+21 return — that value only exists for evaluation, afterward.
          </p>
          <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
            <div className="rounded-md border border-border bg-secondary/30 px-3 py-2.5">
              <dt className="text-[11px] uppercase tracking-wide text-muted-foreground">Forecasting calendar</dt>
              <dd className="mt-0.5 font-mono text-sm font-semibold text-foreground">47 formation dates</dd>
              <dd className="font-mono text-xs text-muted-foreground">2022-02-25 → 2026-01-02</dd>
            </div>
            <div className="rounded-md border border-border bg-secondary/30 px-3 py-2.5">
              <dt className="text-[11px] uppercase tracking-wide text-muted-foreground">Portfolio calendar</dt>
              <dd className="mt-0.5 font-mono text-sm font-semibold text-foreground">46 formation dates</dd>
              <dd className="font-mono text-xs text-muted-foreground">2022-03-28 → 2026-01-02</dd>
            </div>
          </div>
          <p>
            Why 46, not 47: the first forecasting formation (2022-02-25) has only 252 prior trading-session TRI levels
            available — one short of the 253 levels needed to compute the 252 one-session covariance returns
            portfolio construction requires. Rather than shifting the date or padding the missing history, that one
            formation is excluded from portfolio construction entirely; it remains a fully valid forecasting date on{" "}
            <Link to="/forecasts" className="underline underline-offset-2 hover:text-foreground">
              Forecast Rankings
            </Link>
            .
          </p>
        </CardContent>
      </Card>

      {/* D. Modeling */}
      <Card>
        <CardHeader>
          <CardTitle>Modeling</CardTitle>
          <CardDescription>Two pooled models, one fixed ensemble — see Model Comparison for the full walk-forward evaluation.</CardDescription>
        </CardHeader>
        <CardContent className="flex flex-col gap-4 text-sm text-muted-foreground">
          <div>
            <h3 className="text-sm font-semibold text-foreground">XGBoost</h3>
            <p>
              One pooled model across all 50 stocks, trained walk-forward on five features: VIX, the US 10-year
              yield, 3-month momentum, 6-month momentum, and 20-day realized volatility.
            </p>
          </div>
          <div>
            <h3 className="text-sm font-semibold text-foreground">LSTM</h3>
            <p>
              One pooled LSTM reading a 60-session sequence of 13 OHLCV/technical features per stock — no macro,
              static, or ticker-identity features. Trained walk-forward, same as XGBoost.
            </p>
            <details className="mt-1.5 text-xs">
              <summary className="cursor-pointer select-none text-muted-foreground underline underline-offset-2 hover:text-foreground">
                Technical detail
              </summary>
              <div className="mt-1.5 flex flex-col gap-1 font-mono text-muted-foreground/80">
                <span>hidden_size=32, num_layers=1, dropout=0.0</span>
                <span>optimizer=Adam, lr=1e-3, batch_size=64</span>
                <span>max_epochs=50, early-stopping patience=5, seed=42</span>
                <span>
                  The 252-session training window bounds a sequence sample's own end date, not the depth of its
                  60-session input — that input may reach earlier than the window when the history genuinely
                  exists, never fabricated or padded.
                </span>
              </div>
            </details>
          </div>
          <div>
            <h3 className="text-sm font-semibold text-foreground">Ensemble</h3>
            <p>
              Exact arithmetic mean: <span className="font-mono">ensemble_pred = 0.5 · XGB_pred + 0.5 · LSTM_pred</span>.
              This weighting was frozen in advance — never tuned or reweighted after observing walk-forward
              results.
            </p>
          </div>
          <p className="rounded-md border border-border bg-secondary/30 px-3 py-2.5">
            Standalone forecast performance was weak/mixed: the Historical Mean baseline outperformed both XGBoost
            and the ensemble on MAE, RMSE, and directional accuracy in the frozen walk-forward evaluation. Weak
            standalone regression accuracy does not automatically mean zero cross-sectional usefulness — the
            portfolio-construction evidence below tests that question separately.{" "}
            <Link to="/models" className="underline underline-offset-2 hover:text-foreground">
              See the full Model Comparison
            </Link>
            .
          </p>
        </CardContent>
      </Card>

      {/* E. Portfolio research */}
      <Card>
        <CardHeader>
          <CardTitle>Portfolio research</CardTitle>
          <CardDescription>Expected returns, historical covariance, and constrained optimization.</CardDescription>
        </CardHeader>
        <CardContent className="flex flex-col gap-4 text-sm text-muted-foreground">
          <div>
            <h3 className="text-sm font-semibold text-foreground">Expected returns</h3>
            <p>
              The frozen ensemble prediction becomes <span className="font-mono">mu_21</span>, the expected
              21-session return vector used by the two Max-Sharpe strategies only. Min-Vol does not use expected
              returns in its objective at all — it minimizes predicted variance alone.
            </p>
          </div>
          <div>
            <h3 className="text-sm font-semibold text-foreground">Covariance</h3>
            <p>
              252 sessions of historical realized TRI returns as of the formation date, estimated as Sample
              covariance and separately as Ledoit-Wolf shrinkage, then scaled to the 21-session horizon:{" "}
              <span className="font-mono">Σ_21 = 21 · Σ_session</span>. Ledoit-Wolf improved covariance
              conditioning in the historical experiment, but realized portfolio behavior was nearly identical to
              Sample — shrinkage is not shown anywhere as having produced better returns.
            </p>
            <details className="mt-1.5 text-xs">
              <summary className="cursor-pointer select-none text-muted-foreground underline underline-offset-2 hover:text-foreground">
                Risk-free rate
              </summary>
              <p className="mt-1.5 font-mono text-muted-foreground/80">
                rf_21 = ((USGG10YR / 100) / 252) · 21 — used only inside the Max-Sharpe objective.
              </p>
            </details>
          </div>
          <div>
            <h3 className="text-sm font-semibold text-foreground">Constraints &amp; strategies</h3>
            <p>
              All five strategies are long-only and fully invested with a hard 10% maximum single-name weight —
              constrained portfolios, never unconstrained mean-variance optimization. The cap frequently bound for
              the optimized strategies. Strategies: Sample Min-Vol, Sample Max-Sharpe, Ledoit-Wolf Min-Vol,
              Ledoit-Wolf Max-Sharpe, and Equal Weight (exact 1/50 per stock, never optimized).
            </p>
            <details className="mt-1.5 text-xs">
              <summary className="cursor-pointer select-none text-muted-foreground underline underline-offset-2 hover:text-foreground">
                Engineering integrity: numerical robustness
              </summary>
              <p className="mt-1.5 text-muted-foreground/80">
                The optimizer boundary is made C-contiguous to eliminate memory-layout-sensitive SLSQP behavior.
                On the narrow, specific "positive directional derivative for linesearch" solver failure only, a
                single deterministic retry re-initializes from the corresponding Min-Vol weights (same objective,
                constraints, and data) — never substituting Min-Vol as a fallback Max-Sharpe result. Every official
                construction date ultimately solved successfully. Methodology itself was never changed in response
                to results.
              </p>
            </details>
          </div>
          <div>
            <h3 className="text-sm font-semibold text-foreground">Risk decomposition</h3>
            <p>
              Portfolio volatility <span className="font-mono">σ_p = √(w′Σw)</span> decomposes additively into each
              asset's component contribution <span className="font-mono">RC_i = w_i · (Σw)_i / σ_p</span>, which sum
              exactly to <span className="font-mono">σ_p</span>. Because covariance can contain negative
              cross-asset terms, an individual contribution can be negative — a genuine diversifying effect,
              preserved as-is, never floored at zero. Equal Weight is never constructed from a covariance estimate;
              Sample/Ledoit-Wolf is offered there purely as an analysis lens on its fixed weights.
            </p>
          </div>
          <div className="flex gap-4 text-xs">
            <Link to="/portfolio" className="underline underline-offset-2 hover:text-foreground">
              Portfolio Construction
            </Link>
            <Link to="/frontier" className="underline underline-offset-2 hover:text-foreground">
              Efficient Frontier
            </Link>
            <Link to="/risk" className="underline underline-offset-2 hover:text-foreground">
              Risk Analytics
            </Link>
          </div>
        </CardContent>
      </Card>

      {/* F. Evaluation */}
      <Card>
        <CardHeader>
          <CardTitle>Historical evaluation</CardTitle>
          <CardDescription>46 sequential, non-overlapping 21-session periods — historical walk-forward evidence only.</CardDescription>
        </CardHeader>
        <CardContent className="flex flex-col gap-3 text-sm text-muted-foreground">
          <p>
            Realized stock return: <span className="font-mono">TRI(T+21) / TRI(T) − 1</span>. Each period's
            portfolio return uses the weights already frozen at formation time — never re-weighted with hindsight.
            Cumulative evidence compounds sequentially,{" "}
            <span className="font-mono">product_t(1 + r_t) − 1</span>, with no daily interpolation and no
            annualization of the reported figures. Turnover is{" "}
            <span className="font-mono">0.5 · Σ|w_T − w_(T−1)|</span>; each strategy's own first formation has
            genuinely undefined turnover (no prior portfolio exists) and is never shown as zero — Equal Weight's
            later turnover is truthfully zero, by construction.
          </p>
          <p>
            The benchmark is the official Bloomberg S&amp;P 500 Total Return Index (SPXT Index, PX_LAST):{" "}
            <span className="font-mono">SPXT(T+21) / SPXT(T) − 1</span>. SPXT is evaluation-only — it is never used
            in portfolio construction, never a predictive feature, and never a risk coordinate on the Efficient
            Frontier.
          </p>
          <p className="rounded-md border border-border bg-secondary/30 px-3 py-2.5">
            <strong className="text-foreground">Historical walk-forward evidence</strong> from this sample: the
            Max-Sharpe variants compounded to roughly 130% with ~64% average turnover and a ~67% positive-period
            rate; Equal Weight compounded to roughly 92%; SPXT to roughly 60%; the Min-Vol variants compounded to
            roughly 45-46% with substantially lower (~3.9%) period volatility. No transaction costs or slippage
            are modeled, and the hard 10% cap materially shaped the optimized portfolios. This is historical
            evidence from a walk-forward sample, not a guarantee of future performance — see{" "}
            <Link to="/backtest" className="underline underline-offset-2 hover:text-foreground">
              Historical Evidence
            </Link>{" "}
            for the full period-by-period record.
          </p>
        </CardContent>
      </Card>

      {/* G. Sealed holdout */}
      <Card className="border-[var(--color-border-strong)]">
        <CardHeader>
          <CardTitle>Sealed March 2026 holdout</CardTitle>
          <CardDescription>The genuine out-of-sample generalization test — not yet run.</CardDescription>
        </CardHeader>
        <CardContent className="flex flex-col gap-2 text-sm text-muted-foreground">
          <p className="inline-flex w-fit items-center rounded-full border border-[var(--color-border-strong)] bg-secondary/50 px-3 py-1 text-xs font-semibold uppercase tracking-wide text-foreground">
            Sealed — not yet evaluated
          </p>
          <p>
            March 2026 was deliberately withheld from every step above — data collection, feature design, model
            selection, ensemble weighting, covariance choice, and optimizer methodology were all frozen before
            this data was ever consulted. Notably, the historical walk-forward evidence above was itself produced
            from the same pre-March history the models were built on — it is not a fresh out-of-sample test. March
            2026 is that test.
          </p>
          <p>
            The protocol: evaluate the already-frozen pipeline against March exactly once, with no retuning
            regardless of outcome, and report the result honestly as a single sealed case study — not a general
            performance claim, positive or negative.
          </p>
        </CardContent>
      </Card>

      {/* H. Limitations */}
      <Card>
        <CardHeader>
          <CardTitle>Limitations</CardTitle>
        </CardHeader>
        <CardContent>
          <ul className="grid grid-cols-1 gap-x-6 gap-y-1.5 text-sm text-muted-foreground sm:grid-cols-2">
            {LIMITATIONS.map((item) => (
              <li key={item} className="flex items-start gap-2">
                <span aria-hidden="true" className="mt-1.5 h-1 w-1 shrink-0 rounded-full bg-current" />
                {item}
              </li>
            ))}
          </ul>
        </CardContent>
      </Card>

      {/* I. Research integrity — frozen vs. pending, and what RiskFecta does not claim */}
      <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
        <Card>
          <CardHeader>
            <CardTitle>What is frozen</CardTitle>
          </CardHeader>
          <CardContent className="flex flex-col gap-3">
            <ul className="flex flex-col gap-1.5 text-sm text-muted-foreground">
              {FROZEN_ITEMS.map((item) => (
                <li key={item} className="flex items-start gap-2">
                  <span aria-hidden="true" className="mt-1.5 h-1 w-1 shrink-0 rounded-full bg-current" />
                  {item}
                </li>
              ))}
            </ul>
            <div className="border-t border-border pt-2 text-sm">
              <span className="font-semibold text-foreground">Pending:</span>{" "}
              <span className="text-muted-foreground">the sealed March 2026 outcome.</span>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>RiskFecta does not claim</CardTitle>
          </CardHeader>
          <CardContent>
            <ul className="flex flex-col gap-1.5 text-sm text-muted-foreground">
              {DOES_NOT_CLAIM.map((item) => (
                <li key={item} className="flex items-start gap-2">
                  <span aria-hidden="true" className="mt-1.5 h-1 w-1 shrink-0 rounded-full bg-current" />
                  {item}
                </li>
              ))}
            </ul>
          </CardContent>
        </Card>
      </div>
    </motion.div>
  );
}
