# RiskFecta — Phase 0 Plan (Pre-Build)

**Status:** Planning only — no code until you approve.  
**Phase:** 0 — Pre-Build Environment Setup  
**Estimate:** ~3 hrs  
**Source:** Locked Build Plan v1.0 + PRD v1.0

---

## 0. Cursor discipline (locked)

- **Phase numbering:** The PRD/Build Plan numbering is authoritative. Do not renumber phases or invent Phase 10 / Phase A.
- **Before executing:** Ask Cursor to *"Summarize Phase 0 tasks before executing anything"* so it reasons first instead of coding immediately.
- **Always ask:** *"What phase are we in?"* and keep work scoped to that phase.

---

## 1. MVP-A alignment (locked)

MVP-A is complete only when all are true:

| # | Criterion | Build Plan phase that delivers it |
|---|-----------|-----------------------------------|
| 1 | Bloomberg Terminal pull completed | Phase 1 (Step A — campus terminal) |
| 2 | CSV data ingested into PostgreSQL | Phase 1 (Step B — ingest.py) |
| 3 | Rolling-window validation working | Phases 3 + 4 (LSTM + XGBoost loops); integrity audit before run |
| 4 | Ensemble predictions generated | Phase 5 (ensemble.py) |
| 5 | Efficient Frontier computed | Phase 6 (portfolio.py) |
| 6 | Streamlit Cloud deployed | Phase 7 (Streamlit app + deploy) |
| 7 | Demo runs on real Bloomberg data | Phase 7 (data from Phase 1) |

**Constraints:** 40-asset S&P 500 subset for MVP; data must be real Bloomberg export (no synthetic / Yahoo).

---

## 2. Phase mapping (Build Plan = source of truth)

ChatGPT’s list is close; this aligns exactly to the locked Build Plan:

| Your list | Build Plan | Output |
|-----------|------------|--------|
| Phase 0 | **Pre-Build** | Environment + repo + Postgres schema + config |
| Phase 1 | **Phase 1** | Bloomberg pull + CSV → PostgreSQL (ingest.py) → prices_raw + features (partial) |
| Phase 2 | **Phase 2** | Feature engineering (features.py) → features table complete |
| Phase 3 | **Phase 3** | LSTM model (rolling-window) |
| Phase 4 | **Phase 4** | XGBoost model (rolling-window) |
| Phase 5 | **Phase 5** | Ensemble + evaluation |
| Phase 6 | **Phase 6** | Efficient Frontier (MPT optimizer) |
| Phase 7 | **Phase 7** | Streamlit app + Plotly + Streamlit Cloud deployment |
| Phase 8 | **Phase 8** | Tableau (3 dashboards) |
| Phase 9 | **Phase 9** | Tests, README, GitHub polish, portfolio embed |

Note: “Bloomberg → CSV” and “CSV → PostgreSQL” are both in **Phase 1** (Step A + Step B). Phase 2 is feature engineering, not ingestion.

---

## 3. Phase 0 — Pre-Build (mapped in full)

One session, ~3 hrs. Complete entirely before any pipeline code.

---

### Step 1 — GitHub + environment  
**Owner:** Cursor Pro  
**Deliverables**

1. **Repo**
   - Repo exists (RiskFecta); clone locally; open in Cursor Pro.
   - No new repo creation unless you want a fresh one.

2. **requirements.txt** (aligned with PRD/Build Plan)
   - **Workflow:** (1) Let Python 3 / pip resolve: install the core packages below (no versions yet). (2) Run `pip freeze > requirements.txt`. (3) Trim requirements.txt to **top-level packages only** — keep only what the PRD needs; drop transitive deps so the file stays minimal and reproducible.
   - **Core packages (from Build Plan):** torch, xgboost, scikit-learn, psycopg2-binary, pandas, numpy, scipy, plotly, streamlit, pandas-ta; add python-dotenv for .env. Build Plan suggested pins: torch==2.2.0, xgboost==2.0.3, scikit-learn==1.4.0, etc. — after trim, pin top-level only to the versions pip chose (or to these if compatible).
   - Reproducibility: `pip install -r requirements.txt` must succeed in a fresh venv.

3. **Folder structure** (from PRD/Build Plan file paths)
   - `pipeline/` — ingest.py, features.py (Phase 1–2)  
   - `models/` — lstm.py, xgboost_model.py, ensemble.py (Phase 3–5)  
   - `optimizer/` — portfolio.py (Phase 6)  
   - `app/` — main.py + `app/pages/` (Phase 7)  
   - `data/raw/` — Bloomberg CSVs (gitignored)  
   - `tests/` — test_pipeline.py etc. (Phase 9)  
   - `tableau/` — Tableau workbook + optional tableau/docs/ for PDFs (Phase 8)  
   - Root: config.py, requirements.txt, .env (gitignored), .gitignore, README.md  
   - All packages: empty `__init__.py` so they are importable.

4. **.gitignore**
   - Include: `.env`, `data/raw/`, `data/processed/`, `*.pt`, `*.pth`, `__pycache__/`, `.ipynb_checkpoints/`, `venv/`, `.venv/`.
   - **Do not** ignore all `*.csv` globally — that would ignore test fixtures, small exports, and debugging CSVs. Only the raw (and processed) data directories are ignored.

**Check:** Repo structure present; `pip install -r requirements.txt` succeeds; .gitignore in place.

---

### Step 2 — PostgreSQL setup  
**Owner:** Claude Pro (schema) + you (install, .env, run schema)

1. **schema.sql** (Claude Pro to generate from PRD Section 6)
   - **Table 1 — prices_raw:** id, ticker, date, open, high, low, close, volume, total_return_idx; UNIQUE(ticker, date).  
   - **Table 2 — features:** id, ticker, date, rsi_14, macd, macd_signal, bb_upper, bb_lower, volatility_20d, momentum_3m, momentum_6m, beta, mkt_cap_log, sector, div_yield, vix, yield_10y; UNIQUE(ticker, date).  
   - **Table 3 — predictions:** id, ticker, forecast_date, target_date, lstm_pred, xgb_pred, ensemble_pred, actual_return, directional_correct; UNIQUE(ticker, forecast_date).  
   - **Table 4 — portfolios:** id, run_id, ticker, weight, target_return, portfolio_vol, sharpe_ratio, created_at.  
   - **Table 5 — risk_metrics:** id, run_id, metric_name, metric_value, ticker, created_at.  
   - **run_id:** Logical key only (no FK). PRD locks 5 tables; application ensures consistency. Create tables in any order; no optimization_runs table.  
   - Types: per PRD (VARCHAR, NUMERIC, DATE, BOOLEAN, TIMESTAMP, SERIAL, BIGINT).  
   - Comment in schema: “Risk-free rate for Sharpe = USGG10YR / 252 (daily). No T-bill.”

2. **Local PostgreSQL**
   - Install PostgreSQL; create DB (e.g. `riskfecta`).  
   - Test connection (e.g. psql or pgAdmin).

3. **.env** (you create; never commit)
   - `DATABASE_URL=postgresql://localhost/riskfecta` (or your user/host/port).  
   - Application code reads DATABASE_URL only (no hardcoded vendor).

4. **Run schema**
   - Execute schema.sql on that DB.  
   - Verify all 5 tables exist (prices_raw, features, predictions, portfolios, risk_metrics) and UNIQUE constraints are present.

**Check:** All 5 tables created; UNIQUE(ticker, date) on prices_raw and features; UNIQUE(ticker, forecast_date) on predictions.

---

### Step 3 — Config  
**Owner:** Claude Pro to generate; Cursor to place and adjust if needed

1. **config.py** (from Build Plan Step 3)
   - **Ticker universe:** 40–50 S&P 500 symbols, Technology + Financial Services (placeholder list; you’ll replace with actual Bloomberg tickers after Phase 1 if needed).  
   - **Feature column lists:**  
     - LSTM: e.g. OHLCV + RSI, MACD, BB, vol, momentum (names matching schema/features table).  
     - XGBoost: e.g. beta, mkt_cap_log, div_yield, sector (encoded), vix, yield_10y, momentum_3m, momentum_6m, volatility_20d (and any other tabular features from PRD).  
   - **Rolling-window constants (locked):**  
     - TRAIN_WINDOW = 252  
     - STEP = 21  
     - LSTM_SEQ = 60  
     - FORECAST_HORIZON = 30  
   - **Paths:** e.g. data/raw/ for CSVs, any model weight dir if used.

**Check:** config.py imports without error; constants match PRD/Build Plan; feature lists align with schema.

---

### Step 4 — Bloomberg export spec (reference only in Phase 0)

No code here; just a one-page spec so Phase 1 (campus terminal) is unambiguous.

- **Pull 1 — Prices:** PX_LAST, PX_OPEN, PX_HIGH, PX_LOW, PX_VOLUME, EQY_TOTAL_RETURN_INDEX; 40–50 tickers; 5 years daily → e.g. prices_raw.csv.  
- **Pull 2 — Static:** CUR_MKT_CAP, BETA_RAW_OVERRIDABLE, DVD_YLD_IND, GICS_SECTOR_NAME per ticker → e.g. static_fields.csv.  
- **Pull 3 — Macro:** VIX Index, USGG10YR Index, SPX Index (daily) → e.g. macro.csv.  
- **Delivery:** CSVs in data/raw/ (gitignored).

You can add a short `docs/Bloomberg_export_spec.md` or keep this section in this plan only.

---

### Step 5 — Initial sanity checks (optional but recommended)

- Python: `import torch, xgboost, sklearn, psycopg2, pandas, numpy, scipy, plotly, streamlit, pandas_ta` all succeed.  
- DB: from Python, connect with DATABASE_URL from .env; run a trivial query (e.g. list tables or count rows).  
- Config: import config; assert config.TRAIN_WINDOW == 252 and config.STEP == 21.

---

## 4. Phase 0 completion criteria

- [ ] Repo has folder structure above and requirements.txt with pinned versions.  
- [ ] `pip install -r requirements.txt` succeeds.  
- [ ] .gitignore includes .env, data/raw/, data/processed/, *.pt, __pycache__; does **not** globally ignore *.csv (so test fixtures and small CSVs can be committed).  
- [ ] schema.sql created (all 5 tables, UNIQUE constraints, USGG10YR/252 note).  
- [ ] Local PostgreSQL has DB; schema.sql applied; 5 tables verified.  
- [ ] .env exists with DATABASE_URL; not committed.  
- [ ] config.py exists with universe (placeholder), feature lists, TRAIN_WINDOW=252, STEP=21, LSTM_SEQ=60, FORECAST_HORIZON=30.  
- [ ] (Optional) Sanity checks pass.

When all are done, Phase 0 is complete and we proceed to Phase 1 (Bloomberg pull + ingest.py) without skipping steps.

---

### Phase 0 remaining (your action)

- [ ] **PostgreSQL:** Create DB (e.g. `createdb riskfecta` or `CREATE DATABASE riskfecta`), run `psql -d riskfecta -f schema.sql`, verify with `\dt` (prices_raw, features, predictions, portfolios, risk_metrics).
- [ ] **.env:** Create `.env` with `DATABASE_URL=postgresql://user:password@localhost:5432/riskfecta` (never commit).

**Python:** Use **Python 3.12** for this project (pandas-ta and some ML deps not yet compatible with 3.14). Create venv: `python3.12 -m venv venv`, activate, then `pip install -r requirements.txt`.

---

### Phase 1 handoff — ingest reshape (locked)

- **Bloomberg BQL export is typically wide format:** tickers as column headers, dates as rows (often with multi-level headers). The `prices_raw` table expects **long format:** one row per (ticker, date).
- **This reshape is Phase 1’s biggest risk** and is where silent bugs (wrong ticker/date mapping, index misalignment) get introduced.
- **Rule:** Do not let Cursor implement the wide→long reshape in `ingest.py` without **Claude’s audit first**. Claude will provide or audit the reshape logic before any pipeline run.

---

## 5. Data handling (locked)

- **Missing values:** Bloomberg #N/A and missing values remain **NULL** in the database. Do not impute in ingest or in schema.
- **Forward fill (ffill):** Per-ticker ffill happens **only** in `features.py` (Phase 2). **Never** in `ingest.py`. **Never** in `schema.sql`. Raw data in `prices_raw` stays immutable; cleaning is a feature-engineering step only.

---

## 6. Integrity / leakage reminders (for later phases)

- **Rolling-window (Phases 3–4):** Train only on past; predict only forward; no future data in train or features. Get an **Integrity Audit** (e.g. paste loop to Claude Pro) before first training run.  
- **Covariance:** Efficient Frontier uses **realized** returns for covariance, not predicted.  
- **Directional accuracy:** Exclude actual_return == 0 from denominator.  
- **DB:** UNIQUE(ticker, date) and UNIQUE(ticker, forecast_date) in place before any ingestion.

---

*Phase 0 plan complete. No code written until you confirm. Next: execute Step 1 (repo + requirements + folders + .gitignore) when you say go.*
