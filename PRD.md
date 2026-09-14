# RiskFecta — Product Requirements Document (V2)

**Status:** Draft V2 — Phase 0 specification package
**Supersedes:** `PRD_and_buildplan/RiskFecta_PRD.docx` (V1). See [V1 Archive / Supersession](#v1-archive--supersession).
**Authority:** Product scope and intent. Where technical architecture, ML methodology, or execution sequencing conflict with this document, see [`TRD.md`](TRD.md), [`ML_SPEC.md`](ML_SPEC.md), [`BUILD_PLAN.md`](BUILD_PLAN.md) respectively for the authoritative treatment of those concerns — but product scope/intent itself is decided here.

---

## 1. Product Summary

**RiskFecta — Quantitative Portfolio Intelligence Platform.**

RiskFecta is a quantitative portfolio research and construction platform. It ingests Bloomberg market and macro data for a fixed 50-equity universe, produces forward return forecasts from a pooled XGBoost model and a pooled LSTM model, combines them into an equal-weight ensemble, estimates risk (covariance), and constructs long-only portfolios on the efficient frontier via constrained mean-variance optimization. Results are presented through a FastAPI + React/TypeScript web application with Plotly visualizations.

## 2. Problem Statement

**Core question:** Given historical market data and ML forecasts, which assets appear attractive and how should they be combined into a risk-adjusted portfolio?

Institutional-grade portfolio construction tools (Bloomberg PORT, FactSet, Axioma) are expensive and inaccessible outside institutional desks. Public/free alternatives typically fall back to naive historical-average expected returns and rarely combine a rigorous ML forecasting pipeline with disciplined mean-variance optimization and honest, walk-forward out-of-sample evaluation in one reproducible, deployed system.

RiskFecta demonstrates that pipeline end-to-end — from immutable raw Bloomberg exports through leakage-audited ML forecasting to portfolio construction — as a solo-owned research/engineering project.

## 3. User / Audience

- **Primary:** Technical reviewers of the author's work — recruiters, hiring managers, and engineers evaluating quantitative/ML/full-stack ability.
- **Secondary:** The author, as a personal quantitative-research sandbox.

RiskFecta is **not** built for retail investors seeking trade execution, and it is not marketed or positioned as investment advice for any audience.

## 4. Product Positioning

RiskFecta is a **quantitative portfolio research and construction platform**: Bloomberg data in, forward-return forecasts and optimized portfolios out, with every step auditable and walk-forward validated.

It is not a trading system, not a consumer finance app, and not a systemic-risk research tool.

## 5. Differentiation

| | **PlainCents** | **WatStreet Volatility Contagion Explorer** | **RiskFecta** |
|---|---|---|---|
| Domain | Personal finance | Systemic/network risk research | Portfolio return forecasting + construction |
| Data | Bank transactions | ~500-equity volatility panel | Bloomberg 50-equity + macro panel |
| Core technique | Transaction classification, spend forecasting | GAT + LSTM over a dynamic stock graph | Pooled XGBoost + pooled LSTM ensemble |
| Output | Spending/budget views, portfolio tracking | Volatility/contagion forecasts | Return forecasts → mean-variance optimized portfolios |
| Stack | Consumer fintech stack | PyTorch Geometric | Supabase/PostgreSQL, FastAPI, React/TypeScript, Plotly |

RiskFecta is explicitly **not**: a budgeting or transaction-tracking app, a brokerage or execution system, an HFT system, a generic market-news app, or a graph-neural-network systemic-risk research project competing with WatStreet's approach.

## 6. Core User Workflows

1. **Explore the universe.** Browse the 50-equity universe (25 Information Technology, 25 Financials) and their historical market data.
2. **Review forecasts.** See forward-return forecast rankings from XGBoost, LSTM, and the ensemble for the current formation date.
3. **Compare models.** Inspect walk-forward out-of-sample evidence: baselines vs. XGBoost vs. LSTM vs. ensemble, across multiple metrics.
4. **Construct a portfolio.** Choose a point on the efficient frontier (e.g., minimum-volatility, maximum-Sharpe) built from ensemble forecasts and a chosen covariance estimator.
5. **Inspect risk.** Review portfolio-level risk analytics (volatility, concentration, risk contribution) against benchmarks.
6. **Read the methodology.** Understand exactly what is real, what is walk-forward-validated, what is a sealed holdout case study, and what is still a research limitation.

## 7. MVP Scope

- Bloomberg-sourced data for the exact 50-equity universe, OHLCV + total return index + SPX/VIX/USGG10YR macro series + snapshot static fields.
- Normalized PostgreSQL (Supabase-hosted) storage per the 5-table schema.
- Pooled XGBoost and pooled LSTM forward-return forecasters, trained/evaluated via rolling walk-forward temporal splits.
- Equal-weight (50/50) ensemble of the two model branches.
- Baseline comparators (naive/historical-mean, momentum, Ridge regression) alongside XGBoost/LSTM/ensemble.
- Constrained long-only mean-variance optimizer (SciPy) producing minimum-volatility and maximum-Sharpe portfolios and an efficient frontier, benchmarked against equal-weight and SPX.
- FastAPI backend, React + TypeScript frontend, Plotly charts, deployed and publicly reachable.
- A sealed March 2026 holdout case study, evaluated only after all methodology is frozen.
- A methodology/limitations surface that is honest about what is validated vs. exploratory vs. sealed.

## 8. Non-Goals

RiskFecta explicitly does **not** include, for MVP or otherwise unless separately approved:

- Live trading, paper trading, or order execution of any kind.
- Personalized investment advice.
- Consumer budgeting, transaction import/classification, or spend forecasting (PlainCents' domain).
- A dynamic stock graph, GNN/GAT modeling, or systemic/contagion risk research (WatStreet's domain).
- Leverage, short selling, Black-Litterman, risk parity, or robust optimization.
- Ticker-specific (per-symbol) models, or learned ticker embeddings, for the pooled LSTM MVP.
- Dynamically expanding or shrinking the 50-stock universe based on performance.
- Streamlit or Tableau (V1 artifacts — see [V1 Archive](#v1-archive--supersession)).
- A microservices architecture.

## 9. User-Facing Pages / Surfaces

- **Overview / research dashboard** — project status, data coverage, headline honest disclosures.
- **Universe / market data** — the 50-equity universe, sector split, historical price/return views.
- **Forecast rankings** — current-period XGBoost / LSTM / ensemble forecasts, ranked.
- **Model comparison** — walk-forward OOS evidence across baselines, XGBoost, LSTM, ensemble, and metrics.
- **Portfolio construction** — efficient frontier, minimum-volatility and maximum-Sharpe portfolios, weight tables.
- **Efficient frontier** — interactive risk/return frontier chart with benchmark markers (equal-weight, SPX).
- **Risk analytics** — portfolio volatility, concentration, per-asset risk contribution.
- **Historical evidence / backtest** — the walk-forward record model selection was based on.
- **Methodology / limitations** — plain-language explanation of data, methodology, sealed holdout, and known limitations.

## 10. UX Principles

- Prefer clarity over density; a technical reviewer should understand what they are looking at within seconds.
- Every number shown must be traceable to real computed output — never a mock, placeholder, or illustrative figure presented as real.
- Distinguish visually between "historical walk-forward evidence" and "sealed March case study" wherever both appear.
- Label unfinished features as unfinished rather than hiding incompleteness behind polish.

## 11. Truthfulness / Research Disclaimers

- RiskFecta is a research and educational project, not investment advice, and does not claim to be.
- All displayed forecasts, metrics, and portfolios are produced by the actual pipeline against actual Bloomberg-sourced data — never fabricated, hardcoded, or illustrative values presented as real results.
- Any page showing forecasts or portfolio outputs must carry a visible methodology/limitations link.

## 12. Data Freshness Policy

- Bloomberg-sourced history currently ends around 2026-02-27 (exact terminal valid trading session to be confirmed against the normalized trading calendar). The product must not describe this data as live or current beyond that date.
- Until a live/refreshed data pipeline is built and approved, the application must clearly state the as-of date of its underlying data on every page that shows data-derived output.

## 13. Success Criteria

- End-to-end pipeline runs on real Bloomberg data with no synthetic substitutes.
- Walk-forward temporal evaluation (not random splits) governs all model and methodology decisions.
- A deployed, publicly reachable application reflects real repository/database state at every phase (see [TRD.md](TRD.md) for deployment policy).
- The sealed March 2026 holdout is evaluated exactly once, after methodology freeze, and reported honestly regardless of outcome.
- No resume or portfolio claim is made about model performance that isn't backed by actual out-of-sample evidence produced by this pipeline.

## 14. Research-Result Policy

- Historical walk-forward results are the primary statistical evidence for methodology and model-selection decisions.
- The sealed March 2026 holdout is a single case study, not proof of general model quality, and is never used to retune anything.
- Negative or mediocre results are reported as such; this project's integrity depends on not cherry-picking or reframing results after the fact.

## 15. Resume / Portfolio Integrity

- Any metric used publicly (README, resume, portfolio site) must be drawn from actual measured walk-forward or sealed-holdout output — never an aspirational or fabricated figure (e.g., no pre-committed "55%+ directional accuracy" claims; see [ML_SPEC.md](ML_SPEC.md) §20 and §29).
- The final resume-facing metric(s) are chosen only after genuine evaluation exists to choose from.

## 16. Post-MVP Ideas (explicitly separated from MVP)

Not part of MVP scope; may be considered later, each requiring its own explicit approval:

- Expanding-window (vs. rolling-window) evaluation as a research experiment.
- Validation-weighted (vs. equal-weight) ensembling as a research experiment.
- Ticker-identity features (categorical/one-hot for XGBoost; embeddings for LSTM) as an ablation.
- Point-in-time historical static fields (beta, market cap, dividend yield) if such a data source becomes available, enabling their use as predictive features.
- Predicted (vs. historical-realized) covariance.
- Leverage, shorting, Black-Litterman, risk parity, or robust optimization — all require explicit future approval.
- Live/refreshed data ingestion beyond the current Bloomberg export snapshot.

---

## V1 Archive / Supersession

RiskFecta V1 (`PRD_and_buildplan/RiskFecta_PRD.docx`, `RiskFecta_BuildPlan.docx`, `Phase0_Plan.md`) is **historical reference only** and does **not** govern RiskFecta V2. Known V1 decisions that are superseded and must not be inferred as current:

- 30-day / 30-session forecast horizon (V2: **21** trading sessions).
- Streamlit demo and Tableau static dashboards (V2: FastAPI + React/TypeScript + Plotly; no Streamlit, no Tableau).
- Localhost-first PostgreSQL (V2: Supabase-hosted PostgreSQL as the default architecture).
- Predictive use of snapshot static features (beta, market cap, dividend yield) without point-in-time justification (V2: excluded from predictive training; see [ML_SPEC.md](ML_SPEC.md) §10).
- Placeholder/loosely-specified ticker universe (V2: the exact validated 50-symbol Bloomberg export universe, including MRSH, not MMC).

See [`BUILD_PLAN.md`](BUILD_PLAN.md) Phase 0 for the archive mechanics (files moved, not rewritten).
