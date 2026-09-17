import { motion } from "framer-motion";
import { Link } from "react-router-dom";

interface PipelineStep {
  label: string;
  detail: string;
}

const STEPS: PipelineStep[] = [
  { label: "Bloomberg historical data", detail: "OHLCV + total return index, 50-stock universe (25 Information Technology + 25 Financials)." },
  { label: "Feature engineering", detail: "Leakage-safe technical indicators and macro alignment, backward-looking only." },
  { label: "XGBoost + LSTM", detail: "Pooled models, walk-forward evaluated, forecasting the 21-trading-session forward total return." },
  { label: "50/50 ensemble", detail: "Frozen equal-weight arithmetic mean of the two model branches — expected returns." },
  { label: "Sample / Ledoit-Wolf covariance", detail: "Historical realized-return covariance, 252-session lookback window." },
  { label: "Constrained portfolio optimization", detail: "Long-only, fully invested, 10% max single-position weight — Min-Vol and Max-Sharpe." },
  { label: "Historical evaluation", detail: "Walk-forward temporal validation only — never a random train/test split." },
];

const stepMotion = {
  initial: { opacity: 0, y: 16 },
  whileInView: { opacity: 1, y: 0 },
  viewport: { once: true, margin: "-60px" },
};

/** The real RiskFecta research flow, told as a restrained scroll-reveal
 * sequence via framer-motion's `whileInView` — no GSAP/ScrollTrigger
 * needed for a one-direction reveal like this. */
export default function ResearchPipelineStory() {
  return (
    <div className="flex flex-col gap-2">
      {STEPS.map((step, i) => (
        <motion.div
          key={step.label}
          {...stepMotion}
          transition={{ duration: 0.35, delay: i * 0.03 }}
          className="flex items-start gap-4 rounded-md border border-border bg-secondary/20 px-4 py-3"
        >
          <span className="mt-0.5 font-mono text-xs text-muted-foreground">{String(i + 1).padStart(2, "0")}</span>
          <div>
            <p className="text-sm font-semibold text-foreground">{step.label}</p>
            <p className="text-xs text-muted-foreground">{step.detail}</p>
          </div>
        </motion.div>
      ))}
      <p className="pt-1 text-xs text-muted-foreground">
        Full methodology, limitations, and what remains sealed for later evaluation:{" "}
        <Link to="/methodology" className="underline underline-offset-2 hover:text-foreground">
          Methodology &amp; limitations
        </Link>
        .
      </p>
    </div>
  );
}
