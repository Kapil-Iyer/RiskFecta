# RiskFecta — Build Plan (V2)

**Status:** Draft V2 — Phase 0 specification package
**Supersedes:** `PRD_and_buildplan/RiskFecta_BuildPlan.docx`, `PRD_and_buildplan/Phase0_Plan.md` (V1). See [PRD.md — V1 Archive / Supersession](PRD.md#v1-archive--supersession).
**Authority:** Execution sequencing. Product scope in [`PRD.md`](PRD.md); system architecture in [`TRD.md`](TRD.md); ML/quant methodology in [`ML_SPEC.md`](ML_SPEC.md).

Roles (locked, `.cursor/rules/agent-roles.mdc`): **Claude Code implements. Cursor independently verifies.** A phase is not complete until Cursor's verification is green.

This document defines sequencing and gates only. It restates ML/quant methodology figures (`21`, `252`, `60`) only as they govern phase acceptance criteria — the authoritative definitions live in [ML_SPEC.md](ML_SPEC.md).

---

## Phase 0 — Specification + Repository Reset

**Objective:** Establish PRD/TRD/ML_SPEC/Build Plan as the V2 source of truth; archive V1 without deleting it; align `config.py`/`schema.sql`/`requirements.txt` with the approved docs.

**Files/modules:** `PRD.md`, `TRD.md`, `ML_SPEC.md`, `BUILD_PLAN.md` (this package); `PRD_and_buildplan/archive/`; `config.py`, `schema.sql`, `requirements.txt` (alignment edits only, post-approval).

**Implementation tasks:**
- Author the four V2 documents (this task).
- Move V1 reference material into `PRD_and_buildplan/archive/` (content preserved, not rewritten; `.docx` binaries untouched) with a short archive README explaining supersession.
- **After the user approves this documentation package**, and only then:
  - Remove `beta`, `mkt_cap_log`, `div_yield`, `sector` from `config.py:XGBOOST_FEATURE_COLS` (known conflict with [ML_SPEC.md](ML_SPEC.md) §10 — flagged now, fixed later).
  - Confirm `docs/Bloomberg_export_spec.md`'s `DVD_YLD_IND` field name against the raw export's actual `DIVIDEND_INDICATED_YIELD` and correct the doc.
  - Any other config/schema alignment the approved docs surface as necessary.
- Run the cross-document consistency audit (this document's closing section and the response accompanying this package).

**Tests:** None (documentation phase) beyond the consistency audit itself.

**Acceptance criteria:** All four V2 docs exist, are internally consistent (see audit), and are approved by the user. `PRD_and_buildplan/` content preserved under `archive/`.

**Stop/gate criteria:** Do not touch `config.py`/`schema.sql`/`requirements.txt` until the user has explicitly approved the V2 documentation package.

**Must NOT be done yet:** Any ingestion, feature, model, optimizer, API, or frontend code. Any Supabase setup. Any package installation. Any commit.

**Integrity Audit required:** No (documentation phase).

**Handoff to Cursor:** Verify the four documents against this task's locked decisions (spot-check against the checklist in the closing audit section) and confirm no implementation files changed.

---

## Phase 1 — Data Foundation

**Objective:** Get the real Bloomberg export validated, normalized, and loaded into Supabase-hosted PostgreSQL exactly as the exact 50-stock universe.

**Files/modules:** `pipeline/ingest.py`, `pipeline/validate.py` (or equivalent), `data/raw/*.csv` (read-only input), Supabase project + `schema.sql` applied.

**Implementation tasks:**
- Provision the Supabase PostgreSQL project; apply `schema.sql`; set `DATABASE_URL`.
- Implement wide→long reshape of `prices_raw.csv` (per [ML_SPEC.md](ML_SPEC.md) §2, §4) — this reshape is the phase's highest-risk step (ticker/date misalignment) and requires an audit of the reshape logic before any load runs against the real database.
- Implement trading-session filtering (PX_LAST-gated, §3) and NULL-preserving load of genuine missing values.
- Validate and normalize static fields (snapshot) and macro series (`SPX`, `VIX`, `USGG10YR`) per their own normalization rules (see the Macro / Static Persistence Note below for what "load" means for these two — it is **not** a database write in Phase 1).
- Validate the loaded universe is exactly the 50 expected tickers (MRSH present, MMC absent).

**Tests:** Reshape correctness against fixture data; trading-session filter excludes weekend/holiday rows; NULL preservation; `UNIQUE(ticker, date)` integrity; universe-membership assertion (exactly 50, correct symbols).

**Acceptance criteria:** `prices_raw` populated from the real export; exactly 50 distinct tickers present; no synthetic/Yahoo data; row counts consistent with the expected trading-session count over the date range; static fields and macro series validated/normalized per the Macro / Static Persistence Note below.

**Macro / Static Persistence Note (resolved in Phase 1C):** The frozen 5-table `schema.sql` has no raw macro table and no static-fields table — `features` is Phase 3's *engineered-output* table, not a Phase 1 raw-load destination, and macro is date-only (not ticker-scoped) while static fields are a single current-day snapshot that ML_SPEC.md §10 forbids attaching to historical `(ticker, date)` rows. Accordingly, for Phase 1:
- **Macro** (`SPX`, `VIX`, `USGG10YR`) is validated and normalized (`pipeline/normalize.py:normalize_macro`) but **not persisted** as a raw table. It is re-derived from `data/raw/macro.csv` on demand until Phase 3, where it is joined into `features` at contemporaneously-valid dates only (ML_SPEC.md §6).
- **Static snapshot fields** (market cap, beta, dividend yield, sector) are validated and normalized (`pipeline/normalize.py:normalize_static_fields`) but **not** attached to historical `(ticker, date)` rows and **not persisted** to Postgres in Phase 1. They remain descriptive/UI-only metadata (ML_SPEC.md §10) unless a later, explicit schema decision authorizes storing them — never used as historical predictive features regardless of where they end up stored.
- No new database table was, or should be, invented to give these an earlier destination than the frozen schema authorizes.

**Stop/gate criteria:** Do not proceed to Phase 2 until the reshape/load has passed its own Integrity Audit (calendar/session errors, misaligned tickers/dates, duplicate rows — [ML_SPEC.md](ML_SPEC.md) §29).

**Must NOT be done yet:** Feature engineering, targets, any model code, optimizer, FastAPI, React.

**Integrity Audit required:** Yes — calendar/session errors, misaligned tickers/dates, duplicate rows, missing-data leakage (§29, §30).

**Handoff to Cursor:** Verify row counts, universe membership, NULL-handling, and that no calendar-row leaked in as a "trading session."

---

## Phase 2 — Application Skeleton + First Deployment

**Objective:** Stand up a real, deployed FastAPI + React/TypeScript application backed by the real (Phase 1) database, with no fabricated results, and get a public URL live early.

**Files/modules:** `app/` (FastAPI skeleton, routers), `frontend/` (React + TypeScript shell), Plotly wiring, GitHub Actions CI/CD.

**Implementation tasks:**
- FastAPI skeleton with a health endpoint and at least one real DB-backed read endpoint (e.g., universe listing, coverage stats).
- React + TypeScript shell with a basic dataset/coverage view wired to that endpoint, rendered via Plotly where a chart is shown.
- GitHub CI/CD: test + build on push, deploy to the chosen hosting providers ([TRD.md](TRD.md) §13).

**Tests:** API contract smoke tests; frontend build passes; CI pipeline green end-to-end.

**Acceptance criteria:** A public URL is live, showing real dataset coverage/universe information sourced from the Phase 1 database — no forecasts, no backtests, no portfolios yet, and nothing fabricated standing in for them.

**Stop/gate criteria:** The deployed site must not show any prediction, metric, or portfolio output at this phase — those don't exist yet.

**Must NOT be done yet:** Any ML model, feature pipeline beyond what's needed to show raw coverage, optimizer, forecasts.

**Integrity Audit required:** No modeling yet; still confirm no placeholder/fake result is displayed (Website/Truthfulness rule, below).

**Handoff to Cursor:** Confirm the live URL shows only real, current repo/database state and nothing fabricated.

---

## Phase 3 — Feature + Target Pipeline

**Objective:** Build the leakage-safe feature and target pipeline per [ML_SPEC.md](ML_SPEC.md) §6–§8, §10.

**Files/modules:** `pipeline/features.py`, `pipeline/targets.py` (or equivalent).

**Implementation tasks:**
- Trading-session-safe technical features (RSI, MACD, Bollinger, volatility, momentum) computed per ticker, backward-looking only.
- Macro alignment (SPX/VIX/USGG10YR) joined at contemporaneously valid dates only.
- Allowed forward-fill: per-ticker, past-only, explicit max-horizon (§5).
- 21-session TRI forward-return target computation per §8, using valid-session shifts only.
- Exclude static-snapshot predictive features per §10 (fix `config.py:XGBOOST_FEATURE_COLS` here if not already done in Phase 0).

**Tests:** Feature values match hand-computed fixtures; target formula matches §8 exactly on a known small example; no feature/target uses information from `t+1` or later; forward-fill never crosses a ticker boundary or uses future values.

**Acceptance criteria:** `features` table (or equivalent processed store) populated for all valid (ticker, date) pairs with a computable target; leakage checklist (§29) items relevant to features/targets pass.

**Stop/gate criteria:** No model training starts until an Integrity Audit on feature/target alignment passes.

**Must NOT be done yet:** XGBoost, LSTM, ensemble, optimizer.

**Integrity Audit required:** Yes — future leakage, target leakage, calendar/session errors, incorrect shift direction, static snapshot leakage, missing-data leakage, future macro data usage (§29, §30).

**Handoff to Cursor:** Independently recompute the target and at least one feature on a small sample and confirm exact match; confirm no future-dated information appears in any row's features.

---

## Phase 4 — Baselines + XGBoost

**Objective:** Implement and temporally validate the naive/momentum/Ridge baselines and the pooled XGBoost model.

**Files/modules:** `models/baselines.py`, `models/xgboost_model.py`.

**Implementation tasks:**
- Naive/historical-mean, momentum, and Ridge baselines per [ML_SPEC.md](ML_SPEC.md) §15.
- Pooled XGBoost (§9, §16) trained under rolling `TRAIN_WINDOW=252` / `STEP=21` walk-forward splits (§14), with internal temporal validation (§13) driving hyperparameter choice.
- Store OOS forecasts in `predictions` (`xgb_pred`).

**Tests:** Rolling-split boundaries are strictly chronological with no overlap; hyperparameter selection uses only pre-forecast-date data; forecasts stored per `(ticker, forecast_date)`.

**Acceptance criteria:** Walk-forward OOS forecasts exist for XGBoost and all three baselines across available pre-March history; metrics (§20) computed and stored/reported.

**Stop/gate criteria:** No LSTM work starts until XGBoost's walk-forward loop has passed Integrity Audit.

**Must NOT be done yet:** LSTM, ensemble, optimizer.

**Integrity Audit required:** Yes — train/validation overlap, scaler leakage, future leakage (§29, §30).

**Handoff to Cursor:** Verify no chronological overlap between any training window and the forecast date it produced; verify hyperparameters were never chosen using the evaluation window they're scored against.

---

## Phase 5 — LSTM

**Objective:** Implement and temporally validate the pooled LSTM model.

**Files/modules:** `models/lstm.py`.

**Implementation tasks:**
- Pooled LSTM (§9, §17) over `LSTM_SEQ=60`-valid-session sequences (§18), no ticker embeddings (§10).
- Leakage-safe scaling: fit only on each rolling window's training slice (§22).
- Rolling `TRAIN_WINDOW=252` / `STEP=21` walk-forward splits, internal temporal validation for hyperparameters.
- Store OOS forecasts in `predictions` (`lstm_pred`).

**Tests:** Sequence construction uses exactly 60 valid sessions (not calendar rows); scaler is never fit on validation/evaluation data; no ticker-embedding layer present.

**Acceptance criteria:** Walk-forward OOS forecasts exist for LSTM across available pre-March history; metrics (§20) computed.

**Stop/gate criteria:** No ensemble/comparison work starts until LSTM's walk-forward loop has passed Integrity Audit.

**Must NOT be done yet:** Ensemble, optimizer, covariance experiment.

**Integrity Audit required:** Yes — scaler leakage, future leakage, calendar/session errors (§29, §30).

**Handoff to Cursor:** Verify sequence length and session-counting logic; verify scaler-fit boundaries per rolling window.

---

## Phase 6 — Walk-Forward Comparison + Equal-Weight Ensemble

**Objective:** Produce the historical OOS comparison across baselines/XGBoost/LSTM/ensemble and the production 50/50 ensemble.

**Files/modules:** `models/ensemble.py`, evaluation/reporting module.

**Implementation tasks:**
- Compute `ensemble_pred = 0.5*xgb_pred + 0.5*lstm_pred` per §19 (simple mean only — no validation-weighted ensembling).
- Aggregate metrics (§20) across all models/baselines over the full pre-March walk-forward history; produce evidence tables/forecast rankings.
- No sealed March data touched in this phase.

**Tests:** Ensemble arithmetic is exactly the 50/50 mean; evaluation aggregation matches per-period metrics summed/averaged correctly; March data is absent from any input to this phase.

**Acceptance criteria:** A complete historical walk-forward comparison table exists across all baselines, XGBoost, LSTM, and ensemble, with metrics per §20; forecast rankings are reproducible from stored `predictions` rows.

**Stop/gate criteria:** Do not begin optimizer work until this comparison is complete and reviewed.

**Must NOT be done yet:** Optimizer, covariance experiment, March evaluation.

**Integrity Audit required:** Yes — improper March usage (confirm absence), duplicate rows, benchmark timing issues (§29, §30).

**Handoff to Cursor:** Confirm the ensemble formula is exactly 50/50 with no learned weighting; confirm March data appears nowhere in this phase's inputs or outputs.

---

## Phase 7 — Portfolio Optimization

**Objective:** Build the covariance experiment and the constrained mean-variance optimizer using ensemble forecasts as expected returns.

**Files/modules:** `optimizer/covariance.py`, `optimizer/portfolio.py`.

**Implementation tasks:**
- Expected returns from ensemble forecasts (§24) — never future realized returns.
- Covariance experiment: sample vs. Ledoit-Wolf (§23), evaluated on historical walk-forward evidence (stability, realized risk/risk-adjusted outcomes, turnover if measured); select production method from evidence, not preselection.
- **Decision gate resolved here:** the exact maximum single-position weight constraint (§26) is chosen and documented in this phase, with its rationale (evidence-based, not invented) — not left as "0.20" or any other unjustified default.
- Long-only, fully-invested constrained mean-variance optimization (SciPy): minimum-volatility, maximum-Sharpe, efficient frontier, equal-weight comparison (§26).
- Risk-free rate per §25 (`(USGG10YR/100)/252`) wired into Sharpe computation.

**Tests:** Covariance matrix is PSD and computed only from realized historical returns as of each formation date; optimizer respects long-only/fully-invested/max-weight constraints exactly; Sharpe uses the correctly unit-normalized risk-free rate.

**Acceptance criteria:** Efficient frontier, min-vol, and max-Sharpe portfolios computed across the historical walk-forward evaluation periods, compared against equal-weight and SPXT benchmarks (§27); covariance-method choice is documented with its supporting evidence; max-weight constraint is documented with its supporting rationale. The Phase 7 portfolio experiment covers the **46** covariance-eligible formations of the Phase 4–6 47-date forecasting calendar (2022-02-25 excluded for insufficient covariance history — see [ML_SPEC.md](ML_SPEC.md) §23; the underlying 47-date `predictions` calendar is unchanged).

**Stop/gate criteria:** No leverage/shorting/Black-Litterman/risk parity/robust optimization without separate explicit approval.

**Must NOT be done yet:** March evaluation.

**Integrity Audit required:** Yes — accidental future covariance information, benchmark timing issues, improper March usage (confirm absence) (§29, §30).

**Handoff to Cursor:** Verify covariance and optimizer inputs never include a return realized after the relevant formation date; verify the risk-free-rate unit conversion; verify constraint satisfaction on every produced portfolio.

**Results (frozen, executed once, 2026-09-16 — Cursor Post-Run Integrity Audit: PASS):**

The official Phase 7 experiment ran across the **46 covariance-eligible formations** (2022-03-28 through 2026-01-02; 2022-02-25 excluded per above). All figures below are historical evidence from **46 sequential, non-overlapping 21-trading-session periods** — **not annualized**, no risk-adjusted (Sharpe-style) aggregate computed, no transaction costs modeled.

| Strategy | Mean 21-session return | Std | Hit rate | Mean turnover | Avg max weight | Cumulative return |
|---|---|---|---|---|---|---|
| SAMPLE_MINVOL | 0.89% | 3.89% | 60.9% | 0.084 | 0.100 | 45.4% |
| SAMPLE_MAXSHARPE | 2.00% | 5.88% | 67.4% | 0.636 | 0.100 | 130.6% |
| LW_MINVOL | 0.90% | 3.88% | 60.9% | 0.081 | 0.100 | 45.9% |
| LW_MAXSHARPE | 2.00% | 5.88% | 67.4% | 0.636 | 0.100 | 130.3% |
| EQUAL_WEIGHT | 1.57% | 5.45% | 60.9% | 0.000 | 0.020 | 91.6% |
| SPXT (benchmark) | 1.11% | 4.18% | 67.4% | — | — | 60.0% |

Interpretation, per the frozen methodology:

- `SAMPLE_MAXSHARPE`/`LW_MAXSHARPE` use the ML ensemble expected-return signal (`mu_21`); `SAMPLE_MINVOL`/`LW_MINVOL` are **mu-independent by construction** (no expected-return input at all).
- The 10% per-stock cap (`MAX_WEIGHT`) was **frequently binding** across all four optimized strategies (average max observed weight ≈ 0.100 in every case).
- `EQUAL_WEIGHT` turnover is ≈0 by construction (weights stay fixed at 1/50 every period), not a modeling result.
- Ledoit-Wolf substantially improved covariance conditioning versus Sample (mean condition number ≈327 vs. ≈694) but produced **very similar realized portfolio outcomes** to Sample covariance — the shrinkage/stability benefit did not translate into a materially different result in this sample.
- `MAXSHARPE` produced stronger historical raw returns than Equal Weight/SPXT in this sample, at substantially higher turnover and somewhat higher volatility; `MINVOL` underperformed both benchmarks.
- No transaction costs are modeled — these are **not** live/investable returns, and this is **not** evidence that "ML works" in general; the preferred reading is **promising historical evidence in this one sample**. Phase 6's standalone forecasting metrics were themselves weak (see Phase 6 results above), which tempers how much weight this single portfolio-level result should carry on its own.
- The sealed March 2026 holdout (Phase 9) remains untouched and is the actual out-of-sample generalization test — this Phase 7 result was itself produced from the same pre-March history the models were built on, not a fresh sample.

Full per-strategy statistics (median/min/max/median-turnover/max-turnover/max-observed-weight) and the Sample-vs-Ledoit-Wolf covariance diagnostics (eigenvalues, PSD status) are recorded in the Phase 7B execution report; persisted in `portfolios` (11,500 rows) and `risk_metrics` (685 rows), `run_id` prefix `p7bv1_`.

---

## Phase 8 — Full Research Dashboard

**Objective:** Complete the recruiter-facing research dashboard across all PRD user-facing surfaces.

**Files/modules:** `frontend/` pages for forecasts, model comparison, portfolio construction, efficient frontier, risk analytics, historical evidence, methodology/limitations ([PRD.md](PRD.md) §9).

**Implementation tasks:**
- Wire every remaining PRD surface to real backend data produced by Phases 3–7.
- Methodology/limitations page states data freshness, what's walk-forward-validated vs. sealed-holdout vs. exploratory, and known limitations ([ML_SPEC.md](ML_SPEC.md) §32).

**Tests:** Frontend integration tests against the real API; visual/manual QA against the PRD's UX principles.

**Acceptance criteria:** All PRD-listed pages are live, backed by real data, with no placeholder content.

**Stop/gate criteria:** March data/results still do not appear anywhere yet.

**Must NOT be done yet:** March evaluation results.

**Integrity Audit required:** No new modeling; confirm no March leakage into anything shown (§29).

**Handoff to Cursor:** Confirm every page's numbers trace to a real backend computation; confirm no March-derived content appears.

---

## Phase 9 — Sealed March 2026 Evaluation

**Objective:** Execute the one-time sealed March 2026 holdout case study.

**Files/modules:** Evaluation script/module reading frozen model artifacts + freshly obtained March realized data.

**Implementation tasks:**
- **Confirm freeze first:** feature set, model definitions, hyperparameters, ensemble, covariance choice, optimizer methodology, and evaluation procedure are all frozen from Phases 3–7, with no pending changes.
- Obtain the next 21 valid trading-session realized outcomes after the training-data terminal date (~2026-02-27), documenting the data source used.
- Evaluate forecasts and portfolio outcomes against this sealed data — exactly once, no retuning afterward regardless of result.
- Report results honestly, clearly labeled as a single sealed case study, not a general performance claim.

**Tests:** Confirm no March data appears in any artifact produced by Phases 3–7 (re-run the relevant leakage checks against final frozen artifacts before evaluation).

**Acceptance criteria:** A documented, one-time March evaluation report exists, clearly labeled as such, with no subsequent methodology change made in response to its result.

**Stop/gate criteria:** If freeze confirmation fails (something in Phases 3–7 is still unsettled), stop and resolve that first — do not evaluate March under an unfrozen methodology.

**Must NOT be done yet:** Nothing after this depends on holding back further ML work — this is the terminal ML evaluation step.

**Integrity Audit required:** Yes — improper March usage is the central risk of this entire phase (§29, §30), audited both before evaluation (freeze confirmed) and after (no retuning occurred).

**Handoff to Cursor:** Independently confirm the freeze predates the March data pull, and that no commit after the March evaluation touches model/feature/optimizer methodology.

---

## Phase 10 — Hardening + Release

**Objective:** Finalize tests, CI/CD, error handling, production/demo state policy, README, and a truthful resume-facing release.

**Files/modules:** `tests/` (full suite), CI/CD config, `README.md`, demo assets.

**Implementation tasks:**
- Full test suite across pipeline/models/optimizer/API.
- CI/CD audit against [TRD.md](TRD.md) §14.
- Error handling and production/demo-state policy audit ([TRD.md](TRD.md) §16, §20).
- README finalized to reflect actual V2 state (superseding the current V1-notice README).
- Screenshots/demo video of the deployed application.
- Resume metrics drawn only from actual Phase 4–9 evidence — nothing aspirational.

**Tests:** Full `pytest` suite green in CI; no skipped leakage checks.

**Acceptance criteria:** README, tests, CI/CD, and public deployment all reflect the real, final V2 state; final Integrity Audit passes across the whole pipeline.

**Stop/gate criteria:** None beyond the final Integrity Audit passing.

**Must NOT be done yet:** N/A — final phase.

**Integrity Audit required:** Yes — full-pipeline final pass across every item in [ML_SPEC.md](ML_SPEC.md) §29.

**Handoff to Cursor:** Final end-to-end verification and sign-off before calling the project release-ready.

---

## Website / Truthfulness Rule (applies to every phase from Phase 2 onward)

Every deployment must reflect real repo/data/database state. It may show dataset coverage, universe, historical market views, project methodology, and development status. It must **never** show fabricated model predictions, placeholder metrics presented as real, fake backtest results, or fake portfolio recommendations. Unfinished research features are clearly labeled as unfinished. Bloomberg history ends around 2026-02-27 and must never be described as live/current beyond that date.

## Integrity Audit Rule (summary)

Before ML training loops, temporal rolling/walk-forward loops, feature/target alignment logic, covariance-selection research, portfolio backtests, or the sealed March evaluation are run or trusted, an explicit Integrity Audit (per [ML_SPEC.md](ML_SPEC.md) §29–§30) is required. **Claude Code implements. Cursor independently verifies.** Only after green verification is a phase considered complete.
