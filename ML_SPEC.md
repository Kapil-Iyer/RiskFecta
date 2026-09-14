# RiskFecta — ML / Quantitative Methodology Specification (V2)

**Status:** Draft V2 — Phase 0 specification package
**Authority:** ML/quant research methodology. Product scope lives in [`PRD.md`](PRD.md); system architecture in [`TRD.md`](TRD.md); execution sequencing in [`BUILD_PLAN.md`](BUILD_PLAN.md).

This is the most rigorous document in the package. Where an exact implementation detail is not yet justified by evidence, it is marked as a **decision gate** for a later Build Plan phase — never invented or fabricated here.

---

## 1. Research Objective

Given historical Bloomberg market and macro data for a fixed 50-equity universe, forecast each equity's 21-trading-session forward total return, combine forecasts into a risk-adjusted long-only portfolio via mean-variance optimization, and evaluate the entire methodology honestly under walk-forward temporal validation, culminating in one sealed out-of-sample case study (March 2026).

## 2. Dataset / Universe

- **Source:** Bloomberg Terminal exports (BQL/Excel), `data/raw/` (gitignored, immutable).
- **Universe:** exactly 50 equities — 25 Information Technology + 25 Financials — as validated against the actual raw exports (`config.py:TICKER_UNIVERSE`). The universe **includes MRSH and does not include MMC**. The raw export is authoritative; symbols are never substituted from memory or external assumption.
- **History:** approximately 5 years, ending around **2026-02-27** (exact terminal valid trading session confirmed against the normalized trading calendar, not assumed).
- **Fields:**
  - Daily price panel: `PX_OPEN`, `PX_HIGH`, `PX_LOW`, `PX_LAST`, `PX_VOLUME`, `TOTAL_RETURN_INDEX`, per ticker.
  - Macro daily series: `SPX` (`PX_LAST`), `VIX` (`PX_LAST`), `USGG10YR` (`PX_LAST`).
  - Static snapshot fields (single point-in-time pull, not historical): `CUR_MKT_CAP`, `BETA_RAW_OVERRIDABLE`, `DIVIDEND_INDICATED_YIELD` (Bloomberg field name confirmed from the raw export as `DIVIDEND_INDICATED_YIELD`; `docs/Bloomberg_export_spec.md`'s `DVD_YLD_IND` is a stale field name and is corrected only when that doc is revised in Phase 0/1 implementation — not in this documentation task), `GICS_SECTOR_NAME`.
- The raw prices export is in **wide** Bloomberg panel format (repeated per-ticker column blocks with a shared `DATES` column); normalization reshapes this to **long** `(ticker, date, field)` rows before any modeling.

## 3. Trading-Session Definition

- A valid equity trading session for ticker *i* is a row where **`PX_LAST` (close)** is a genuine observed value — not `#N/A` and not a weekend/holiday placeholder.
- **`TOTAL_RETURN_INDEX` is never used to determine session validity.** It may remain populated (e.g., held flat or carried forward by Bloomberg) on non-trading calendar rows, so using it as a session gate would silently admit invalid rows.
- `21`, `60`, `252`, and every `t+21` reference in this document and in `config.py` refer to **valid trading sessions**, never raw calendar rows. Shifting by 21 raw calendar rows is a defined leakage/correctness bug (see §29).

## 4. Normalization Rules

1. Reshape wide Bloomberg export → long `(ticker, date, field)` rows.
2. Apply the trading-session filter (§3): rows without a genuine `PX_LAST` are excluded from normalized market observations before they reach `prices_raw`.
3. Genuine Bloomberg missing values (`#N/A` on an otherwise valid trading session) are preserved as SQL `NULL` — never fabricated as zero, never silently imputed at this stage.
4. No forward-fill during ingestion/normalization (see §5).
5. Static snapshot fields are loaded once and are not fabricated as if they were historical time series.

## 5. Missing-Data Policy

- Raw Bloomberg files are immutable; missing-data handling never touches `data/raw/`.
- Ingestion/normalization: preserve genuine missing values as NULL; exclude non-trading placeholder rows.
- Forward-filling, where used, happens only in feature engineering (`pipeline/features.py`), strictly per-ticker, using only past values, never future information, with an explicit maximum fill-forward horizon (exact limit is a Phase 3 decision gate, documented in code and in feature engineering tests once implemented — not fabricated here).

## 6. Feature Families

- **Price/technical (per ticker, backward-looking only):** OHLCV, RSI(14), MACD (+ signal), Bollinger Bands, realized volatility (20-session), momentum (3-month, 6-month equivalent in trading sessions).
- **Macro (contemporaneously valid, i.e. known as of the formation date):** SPX level/return, VIX level, USGG10YR yield.
- **Sector:** Sector information is UI/descriptive-only for the MVP and is excluded from predictive model features (see §10).
- **Excluded from predictive features (MVP):** snapshot static fields with no point-in-time history — market cap, beta, dividend yield (§10).

## 7. Leakage Exclusions

No feature or target computation may use information not available as of the relevant formation date `t`. Concretely excluded from predictive use in MVP:

- Snapshot static fields (`CUR_MKT_CAP`, `BETA_RAW_OVERRIDABLE`, `DIVIDEND_INDICATED_YIELD`) — single current-day snapshot, not point-in-time history (§10, §22).
- Any scaler, feature-selection, or hyperparameter-tuning statistic fit on data at or after the formation/evaluation date it is applied to (§22).
- March 2026 realized data, prior to the sealed evaluation (§28).
- `TOTAL_RETURN_INDEX` as a session-validity signal (§3).

## 8. Target Formula

Primary modeling target — **21-trading-session forward total return**, computed from Bloomberg `TOTAL_RETURN_INDEX` (TRI):

```
y(i, t) = TRI(i, t+21) / TRI(i, t) - 1
```

where:
- `i` = ticker, `t` = a valid formation trading session (§3).
- `t+21` = the trading session 21 **valid** sessions after `t` (never 21 raw calendar rows — §3).
- Target type: **raw forward total return** — not SPX-excess return, not cross-sectional rank, as the primary target. Ranking-style metrics (e.g., Spearman rank correlation, top-vs-bottom diagnostics) are still evaluated (§20) but do not replace the primary regression target.

## 9. Pooled-Model Definition

- **One pooled XGBoost model** and **one pooled LSTM model**, each trained on observations pooled across all 50 tickers — not 50 per-ticker models.
- **Parallel, not sequential:** XGBoost and LSTM are independent model branches, trained and evaluated separately, then combined only at the ensemble step (§19).
- Shared model weights (both branches) learn cross-sectionally across the whole universe; per-ticker specialization, if any, must come from features, not from separate per-ticker parameter sets.

## 10. Ticker / Static-Feature Policy

- **No learned ticker embeddings** for the LSTM MVP (Option A, locked).
- **No ordinal ticker ID** (e.g., `AAPL=1, MSFT=2, ...`) as a default XGBoost feature — an arbitrary ordinal encoding implies a false numeric relationship between unrelated symbols.
- Ticker identity as a categorical/one-hot XGBoost feature, or a learned LSTM embedding, is an **optional ablation experiment**, never required MVP behavior, and never silently substituted for the pooled/pooled treatment above.
- **Static snapshot fields** (`static_fields.csv`: market cap, beta, dividend yield) are a **single current-day Bloomberg snapshot**, not point-in-time historical data. Attaching them to 2021–2025 historical rows as though they were known at that time is look-ahead leakage. Resolution:
  - Predictive historical training **excludes** current market cap, current beta, current dividend yield, unless a point-in-time historical version of these fields is separately obtained and documented.
  - These fields may still be stored in the database and shown in the UI as **descriptive current metadata** — never as predictive model inputs without point-in-time justification.
- **Sector (GICS `GICS_SECTOR_NAME`) — resolved for MVP:** sector is treated as **descriptive/UI-only** for the predictive MVP feature set. It is **not** included in the XGBoost or LSTM MVP predictive feature lists. Rationale: the alternative (using sector as a predictive feature under a constant-sector-membership-over-the-sample assumption) is not falsified as unreasonable, but MVP explicitly declines it to avoid carrying an unverified constancy assumption into the primary production models. Using sector as a constant-membership predictive feature is available only as a clearly labeled, separately documented **ablation experiment** in a later phase — not a default or a "maybe." `config.py`'s current `XGBOOST_FEATURE_COLS` including `sector` is a known repo conflict (see [TRD.md](TRD.md) §21, [BUILD_PLAN.md](BUILD_PLAN.md) Phase 0) to be corrected only after this documentation package is approved.
- Current stale V1 feature lists (`config.py:XGBOOST_FEATURE_COLS` = `beta, mkt_cap_log, div_yield, sector, ...`) are marked here for removal from the predictive model configuration; that edit itself happens in Phase 0/1 implementation, not in this documentation task.

## 11. Temporal Split Architecture

Four explicitly distinct data roles, never conflated:

- **(A) Training data** — observations used to fit model parameters within a given rolling window.
- **(B) Internal temporal validation** — a held-out-in-time slice used for hyperparameter/methodology selection, always chronologically after the training slice it validates.
- **(C) Historical out-of-sample walk-forward evaluation** — the full rolling sequence of (train → forecast → realize → score) steps across history; this is the primary statistical evidence for model quality (§21).
- **(D) Final sealed March 2026 holdout** — a single, frozen, never-tuned-against evaluation (§28).

No random train/test splitting anywhere in this pipeline. All splits are temporal.

## 12. Rolling Training Window

`TRAIN_WINDOW = 252` trading sessions (config-locked). Rolling window for MVP — not an expanding window. Expanding-window evaluation may later be run as an explicitly approved research experiment, never a silent substitution.

## 13. Internal Temporal Validation

Within each rolling training window (or across a held-back early sub-range of history), a chronologically later slice is reserved for hyperparameter and methodology selection before that window's forecast is produced. Exact validation-slice sizing is a Phase 4/5 implementation decision gate — not fabricated here — but it must always be chronologically after everything it validates and chronologically before the forecast date it supports.

## 14. Historical OOS Walk-Forward Protocol

- Cadence: `FORECAST_HORIZON = 21` trading sessions, `STEP = 21` trading sessions — approximately monthly, non-overlapping portfolio formation/evaluation periods.
- At each formation date `t`: train (or roll forward) on the trailing `TRAIN_WINDOW` valid sessions, forecast `y(i, t)` for all 50 tickers, then — 21 valid sessions later — score against the realized `y(i, t)`.
- This walk-forward sequence, run across all available pre-March history, is the **main statistical evidence** for model comparison, covariance-method selection, and (eventually) hyperparameter/methodology freeze.

## 15. Model Baselines

At minimum:

- **Naive / historical-mean** — each ticker's forecast is its own trailing historical mean 21-session forward return (mechanics: computed only from the same `TRAIN_WINDOW` of history available at `t`, no future information).
- **Momentum-based** — forecast derived from realized trailing momentum (e.g., prior 3-/6-month return) as a simple linear or rank-based signal, not a re-implementation of the technical `momentum_3m`/`momentum_6m` features beyond their direct use as the baseline's forecast driver.
- **Linear/Ridge regression** — a regularized linear model over the same leakage-safe feature set as XGBoost (§6), fit under the same temporal-split discipline.

Followed by:

- **XGBoost** (pooled, §9).
- **LSTM** (pooled, §9, §18).
- **Equal-weight XGBoost + LSTM ensemble** (§19).

Portfolio-level benchmarks (not model baselines, but required comparators for the optimizer/portfolio evaluation, §27): equal-weight 50-stock portfolio, SPX.

Baselines are deliberately simple mechanics — no arbitrarily complicated baseline construction.

## 16. XGBoost Role

- Implemented **before** LSTM (§9, [BUILD_PLAN.md](BUILD_PLAN.md) implementation order) because it provides a strong, inexpensive nonlinear benchmark and de-risks the feature/target pipeline before the more complex sequential model is built.
- Pooled across all 50 tickers; tabular feature set per (ticker, date) per §6/§10.
- Hyperparameters selected only via internal temporal validation (§13, §21); never via random CV.

## 17. LSTM Role

- Pooled across all 50 tickers; sequential feature set per (ticker, date) per §6/§10/§18.
- No ticker embeddings (§10).
- Hyperparameters selected only via internal temporal validation (§13, §21).

## 18. LSTM 60-Session Sequence

`LSTM_SEQ = 60` trading sessions (config-locked): each training/inference example is a 60-valid-session lookback window of features ending at formation date `t`, predicting `y(i, t)` per §8. Sequence construction must respect the trading-session definition (§3) — a 60-session window is 60 valid sessions, not 60 raw calendar rows.

## 19. Equal-Weight Ensemble

MVP production ensemble is a **simple arithmetic mean**:

```
ensemble_pred(i, t) = 0.5 * xgb_pred(i, t) + 0.5 * lstm_pred(i, t)
```

Validation-weighted (or otherwise learned) production ensembling is **not** implemented for MVP. Alternative weighting schemes may be evaluated later strictly as a research experiment, reported separately from the production ensemble, never silently replacing the 50/50 mean.

## 20. Metrics

Evaluated across multiple dimensions, not optimized for a single resume number:

- MAE, RMSE (regression accuracy).
- Directional accuracy (sign match between forecast and realized return; observations with `actual_return == 0` excluded from the denominator).
- Pearson correlation (forecast vs. realized).
- Spearman / rank correlation (forecast vs. realized).
- Cross-sectional ranking usefulness (e.g., forecast rank vs. realized rank, per formation date).
- Top-vs-bottom style ranking diagnostic (e.g., realized return spread between top- and bottom-ranked forecast deciles/quintiles), where statistically appropriate given the 50-name universe.

No target metric threshold (e.g., "55%+ directional accuracy") is fixed anywhere in this document or elsewhere in the V2 package. The final resume-facing metric is chosen only after genuine OOS evaluation exists (see [PRD.md](PRD.md) §13, §15).

## 21. Model-Selection Rules

- Hyperparameters and methodology choices are selected using historical temporal validation (§13, §14) only.
- Once final methodology — feature set, model definitions, hyperparameters, ensemble weights, covariance method, optimizer methodology, evaluation procedure — is frozen using pre-March data, the sealed March holdout (§28) is consulted exactly once, with no retuning afterward.
- Historical walk-forward results are the main statistical evidence for model quality; March is a final sealed case study, not the sole or primary proof.

## 22. Scaler / Preprocessing Fitting Rules

- Any scaler, normalizer, or feature-selection statistic used by a model is fit **only** on data available as of the relevant training window/formation date — never on the internal validation slice, the historical OOS evaluation period it's being scored against, or March.
- Each rolling-window step re-fits (or appropriately rolls) preprocessing statistics from that step's own trailing training data — a single globally-fit scaler across all history is a leakage bug (see §29).

## 23. Covariance Experiment

- Covariance is estimated from **historical realized returns available as of each formation date** — never future realized returns, never predicted returns.
- MVP evaluates two candidate estimators, with **no predeclared winner**:
  - (A) Sample covariance.
  - (B) Ledoit-Wolf shrinkage covariance.
- Selection between (A) and (B) is made from historical walk-forward evidence: numerical stability, portfolio stability, turnover (if measured), realized risk, realized risk-adjusted outcomes, and other defensible stability diagnostics — evaluated in [BUILD_PLAN.md](BUILD_PLAN.md) Phase 7.
- No predicted-covariance model is introduced in MVP.

## 24. Optimizer Inputs

- **Expected returns:** the ensemble forecast (§19) for each ticker at the relevant formation date — never future realized returns.
- **Covariance:** per §23, from historical realized returns as of the formation date — never future-realized returns, never predicted covariance.
- Future realized returns must never enter portfolio construction in any form.

## 25. Risk-Free-Rate Units / Conversion

Bloomberg `USGG10YR` is expected to be quoted in **annual percentage points** (e.g., a value of `4.5` means approximately 4.5% annualized), consistent with the raw macro export (observed values ~1.3–1.4 in early-2021 rows, consistent with percentage-point quoting). During ingestion/validation, this unit convention is explicitly confirmed against the raw data (not merely assumed) before being relied upon downstream.

Given that confirmation:

```
annual_rf_decimal = USGG10YR / 100
session_rf = annual_rf_decimal / 252 = (USGG10YR / 100) / 252
```

This is a modeling approximation (simple annual-to-session division by 252 trading sessions/year), documented as such — not implied to be the only mathematically valid conversion. **`USGG10YR / 252` without the `/100` percent-normalization step is explicitly wrong and must never be used.**

## 26. Portfolio Constraints

Constrained mean-variance optimization via SciPy (SLSQP or equivalent), supporting at minimum:

- Minimum-volatility portfolio.
- Maximum-Sharpe portfolio.
- Efficient frontier (a range of target-return or target-risk points).
- Equal-weight comparison.

MVP constraints: **long-only**, **fully invested** (weights sum to 1), and a **sensible maximum single-position weight** to bound concentration.

**Decision gate:** the exact maximum-weight constraint value is deliberately **not fixed in this document**. It is resolved in [ML_SPEC.md](ML_SPEC.md)/[BUILD_PLAN.md](BUILD_PLAN.md) Phase 7, from evidence (e.g., diversification/turnover/stability behavior across candidate values), not invented here.

Not in MVP: leverage, shorting, Black-Litterman, risk parity, or robust optimization, unless separately and explicitly approved.

## 27. Portfolio-Level Benchmarks

- Equal-weight 50-stock portfolio.
- SPX (total return).

Both are computed over the same historical walk-forward evaluation periods as the optimized portfolios, for like-for-like comparison.

## 28. March Sealed Holdout

- Bloomberg training/history data ends around **2026-02-27**; the exact terminal formation date is derived from the actual valid trading calendar, not assumed.
- After feature set, model definitions, hyperparameters, ensemble, covariance choice, optimizer methodology, and evaluation procedures are **frozen** using only pre-terminal-date data, the next **21 valid trading-session** realized outcomes are obtained as the sealed March 2026 case study.
- March realized observations **never** enter training, internal validation, feature selection, hyperparameter tuning, ensemble-weight determination, covariance-method selection, or optimizer-parameter determination.
- No retuning after observing March performance, regardless of outcome.
- Any data-source change used to obtain March actuals (e.g., a different Bloomberg pull, or any alternate provider used only to confirm realized outcomes) is explicitly documented; alternative-provider data is never mixed into model **training** under any circumstance.

## 29. Leakage Checklist

Applies before any training loop, walk-forward loop, feature/target alignment logic, covariance-selection research, portfolio backtest, or the sealed March evaluation is trusted (see §30 Integrity Audit):

- Future leakage (any feature/target using information not available as of `t`).
- Target leakage (target components leaking into features).
- Scaler leakage (preprocessing fit on validation/evaluation/March data).
- Cross-sectional leakage (information from one ticker/date leaking into another's feature/target at the same or an earlier date).
- Calendar/session errors (raw calendar rows treated as trading sessions; `TOTAL_RETURN_INDEX` used as a session gate instead of `PX_LAST` — §3).
- Incorrect shift direction (e.g., `t+21` accidentally computed as `t-21`, or off-by-one session shifts).
- Train/validation overlap (chronological overlap between training and validation/evaluation slices).
- Improper March usage (March data influencing anything frozen before it — §28).
- Static snapshot leakage (current-day static fields attached to historical rows as if known historically — §10).
- Missing-data leakage (forward-fill using future values, or fabricated non-NULL values standing in for genuine missing data).
- Future macro data usage (macro series value dated after the formation date used as a contemporaneous feature).
- Accidental future covariance information (covariance computed using returns realized after the formation date).
- Benchmark timing issues (SPX/equal-weight benchmark computed over a different, misaligned date range than the portfolio it's compared against).
- Duplicate rows (e.g., duplicate `(ticker, date)` pairs inflating a training set).
- Misaligned tickers/dates (a reshape/join bug attaching ticker A's date-t row to ticker B, or shifting an entire column block by one row).

## 30. Integrity Audit Requirements

Mandatory, before the user runs or trusts: ML training loops, temporal rolling/walk-forward loops, feature/target alignment logic, covariance-selection research, portfolio backtests, or the sealed March evaluation.

Integrity Audit checks include, as applicable, everything in the Leakage Checklist (§29) plus:

- Duplicate rows; misaligned tickers/dates (cross-referenced, not only listed once).
- **`TOTAL_RETURN_INDEX` used as session gate instead of `PX_LAST`** — checked explicitly, every time, as its own line item, not merely implied by the general calendar/session-errors check.
- Improper March usage (re-verified specifically at the point March data is first touched, not only at methodology-freeze time).

Process: **Claude Code implements. Cursor independently verifies.** A phase is not considered complete until Cursor's verification is green. This applies at minimum before Build Plan Phases 3–7 and 9 are trusted (see [BUILD_PLAN.md](BUILD_PLAN.md)).

## 31. Reproducibility

- All modeling code is deterministic given a fixed random seed, a fixed `config.py`, and the immutable raw Bloomberg export — re-running the pipeline against the same inputs reproduces the same `predictions`/`portfolios`/`risk_metrics` rows (up to documented, seed-controlled stochastic elements, e.g. LSTM weight initialization).
- Every walk-forward evaluation run records enough metadata (formation date range, `config.py` constants, code version) to be reconstructed later.

## 32. Limitations

- 50-name universe (25 IT + 25 Financials) — results do not generalize to a broader market or other sectors without separate validation.
- ~5 years of history with a 252-session rolling window means a limited number of independent 21-session walk-forward evaluation periods; statistical power is inherently limited.
- Snapshot static fields (market cap, beta, dividend yield, and — for MVP — sector) are excluded from predictive use precisely because point-in-time historical versions are not available; this is a real information restriction, not merely a convenience choice.
- The March 2026 sealed holdout is one case study, not a statistically powered out-of-sample test by itself.
- Ensemble weighting is fixed at 50/50 by design choice for MVP, not derived from validation performance.
