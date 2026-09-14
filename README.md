# RiskFecta

**Quantitative Portfolio Intelligence Platform**

RiskFecta is a quantitative portfolio research and construction platform. It ingests institutional-grade Bloomberg market and macro data for a fixed 50-equity universe (25 Information Technology + 25 Financials), forecasts **21-trading-session forward total returns** with a pooled **XGBoost** model and a pooled **LSTM** model, combines them into a **50/50 equal-weight ensemble**, and feeds the ensemble forecasts — alongside a historical-realized-return covariance estimate — into a constrained **mean-variance optimizer** to construct long-only portfolios on the **Efficient Frontier**. Results are (once built) presented through a **FastAPI** backend, a **React + TypeScript** frontend, and **Plotly** charts.

RiskFecta is **not** a live trading or order-execution system, and it is **not investment advice**. It is a research and analytics project with strict walk-forward, out-of-sample validation and look-ahead controls.

> **Authoritative specification:** [`PRD.md`](PRD.md) (product), [`TRD.md`](TRD.md) (architecture), [`ML_SPEC.md`](ML_SPEC.md) (ML/quant methodology), [`BUILD_PLAN.md`](BUILD_PLAN.md) (execution sequencing). These four documents govern RiskFecta V2 and supersede everything in `PRD_and_buildplan/archive/` (V1 — historical reference only; see [PRD.md § V1 Archive / Supersession](PRD.md#v1-archive--supersession)).

> **Status:** Phase 1 (Data Foundation) complete. Bloomberg CSV data pull is **done**; `prices_raw` is validated, normalized, and ingested into Supabase-hosted PostgreSQL (62,800 rows — exact 50-stock universe, 1,256 valid trading sessions per ticker). Macro (`SPX`/`VIX`/`USGG10YR`) and static snapshot fields (market cap, beta, dividend yield, sector) are validated and normalized but intentionally **not** persisted as their own database tables in Phase 1 — the frozen 5-table schema has no raw destination for them; see [BUILD_PLAN.md](BUILD_PLAN.md) Phase 1's Macro / Static Persistence Note. **Phase 2A (FastAPI backend skeleton) is implemented**: read-only endpoints over the real Phase 1 database (see [Running the API locally](#running-the-api-locally)). **Phase 2B (React + TypeScript + Plotly frontend) is implemented**: a local dashboard consuming that API — real market coverage, universe browser, and historical price/volume charts, with build status honestly marked "planned" where models/portfolios don't exist yet (see [Running the frontend locally](#running-the-frontend-locally)). Deployment (Phase 2C) has **not** been started, and no model, optimizer, forecast, or portfolio code exists yet — see [Current state vs. roadmap](#current-state-vs-roadmap). **No forecasts, backtests, or portfolio results exist yet, and none are presented as real anywhere in this repository.**

---

## The problem we are solving

Professional portfolio tools (e.g. Bloomberg PORT, FactSet) cost tens of thousands of dollars a year and are out of reach for most retail investors and researchers. Free alternatives typically fall back to naive historical-average expected returns and rarely combine a rigorous ML forecasting pipeline with disciplined mean-variance optimization and honest, walk-forward out-of-sample evaluation in one reproducible, deployed system.

**RiskFecta's goal:** build an accessible, interview-defensible quant research platform that:

1. Sources real Bloomberg data (BQL / Excel export → CSV → Supabase-hosted PostgreSQL).
2. Forecasts 21-trading-session forward total returns with a pooled dual-model ensemble (pooled LSTM + pooled XGBoost, trained/evaluated under temporal walk-forward splits — never random splits).
3. Optimizes long-only portfolios on the Efficient Frontier using ensemble expected returns and historical-realized-return covariance (sample vs. Ledoit-Wolf shrinkage, selected from walk-forward evidence, not preselected).
4. Presents outcomes through a FastAPI + React/TypeScript web application with Plotly visualizations.

## Architecture (target state)

```mermaid
flowchart LR
    subgraph ingest [Phase 1]
        BB[Bloomberg BQL / Excel]
        CSV[CSV exports]
        PG[(Supabase PostgreSQL)]
        BB --> CSV --> PG
    end

    subgraph ml [Phase 3-6]
        LSTM[Pooled LSTM - 60-session sequences]
        XGB[Pooled XGBoost - tabular features]
        ENS[50/50 equal-weight ensemble]
        LSTM --> ENS
        XGB --> ENS
    end

    subgraph opt [Phase 7]
        COV[Covariance: sample vs Ledoit-Wolf]
        MPT[SciPy constrained mean-variance optimizer]
        EF[Efficient Frontier]
        ENS --> MPT
        COV --> MPT
        MPT --> EF
    end

    subgraph app [Phase 2 & 8]
        API[FastAPI]
        FE[React + TypeScript + Plotly]
        EF --> API --> FE
    end

    PG --> LSTM
    PG --> XGB
    PG --> COV
```

| Layer | Role |
|--------|------|
| **Bloomberg Terminal** | Raw OHLCV, total return index, static equity fields, macro (VIX, USGG10YR, SPX) |
| **Python pipeline** | Ingestion, trading-session filtering, backward-looking technical indicators |
| **Pooled LSTM (PyTorch)** | 60-valid-session sequences → 21-session forward total-return forecast |
| **Pooled XGBoost** | Tabular macro/technical features → 21-session forward total-return forecast |
| **Ensemble** | `0.5 × XGBoost + 0.5 × LSTM` → expected return vector (fixed 50/50; not validation-weighted) |
| **Optimizer (SciPy)** | Long-only, fully-invested, constrained Efficient Frontier; covariance from **realized** returns only (sample vs. Ledoit-Wolf, chosen from evidence) |
| **FastAPI** | Thin API layer over the pipeline/model/optimizer modules |
| **React + TypeScript + Plotly** | Research dashboard: universe, forecasts, model comparison, portfolio construction, risk analytics, methodology |

**Validation (locked, see [ML_SPEC.md](ML_SPEC.md)):** rolling window — **252**-trading-session train, **21**-session step, **21**-session forecast horizon, `LSTM_SEQ = 60`. Walk-forward temporal splits only — no random train/test splitting. A single **sealed March 2026 holdout** case study is evaluated once, only after methodology is fully frozen. Risk-free rate for Sharpe: `(USGG10YR / 100) / 252` (percent-normalized annual yield converted to a per-session rate).

---

## Current state vs. roadmap

### What exists today

| Deliverable | Status |
|-------------|--------|
| Repo layout (`pipeline/`, `models/`, `optimizer/`, `app/`, `tests/`) | Done |
| `requirements.txt`, `config.py` (universe, features, rolling constants) | Done |
| `schema.sql` (5 tables: prices, features, predictions, portfolios, risk_metrics) | Done |
| `docs/SETUP.md`, `docs/Bloomberg_export_spec.md` | Done |
| **V2 specification package** — `PRD.md`, `TRD.md`, `ML_SPEC.md`, `BUILD_PLAN.md` | Done (frozen) |
| Bloomberg CSV data pull | **Done** |
| `prices_raw` validation, normalization, and PostgreSQL ingestion (Phase 1A/1B/1C) | **Done** — 62,800 rows, exact 50-stock universe |
| Macro/static validation + normalization (in-memory; not persisted — see Phase 1 note) | **Done** |
| FastAPI backend skeleton (`app/`) — `/health`, `/api/universe`, `/api/prices/{ticker}`, `/api/market/summary` | **Done** (Phase 2A) — read-only, real DB data, no fabricated results |
| React + TypeScript + Plotly frontend (`frontend/`) — market summary, universe browser, price/volume charts | **Done** (Phase 2B) — consumes the real Phase 2A API, no fabricated results |
| Feature engineering, ML models, optimizer | **Planned** — Phases 3–7 |
| Deployment (public URL) | **Planned** — Phase 2C |

No model has been trained, no forecast has been produced, and no portfolio has been optimized. Nothing in this repository presents a fabricated or illustrative result as real.

### Build phases

Aligned with the locked [`BUILD_PLAN.md`](BUILD_PLAN.md):

| Phase | Focus | Key output |
|-------|--------|------------|
| **0 — Specification** | PRD, TRD, ML Spec, Build Plan; V1 archive; config/schema alignment | Complete |
| **1** | Bloomberg validation + ingestion | Complete — `prices_raw` populated (62,800 rows); exact 50-stock universe verified |
| **2A** | FastAPI backend skeleton | Complete — read-only endpoints over the real Phase 1 database |
| **2B** | React + TypeScript + Plotly frontend | Complete — local dashboard consuming the real Phase 2A API |
| **2C** | First deployment | **Next** — public URL showing real dataset coverage only — no fabricated results |
| **3** | Feature + target pipeline | Leakage-safe features; 21-session TRI targets |
| **4** | Baselines + XGBoost | Walk-forward OOS forecasts, pooled XGBoost |
| **5** | LSTM | Walk-forward OOS forecasts, pooled LSTM |
| **6** | Walk-forward comparison + ensemble | Model comparison evidence; 50/50 ensemble |
| **7** | Portfolio optimization | Covariance experiment; Efficient Frontier; min-vol/max-Sharpe |
| **8** | Full research dashboard | All PRD-listed pages live on real data |
| **9** | Sealed March 2026 evaluation | One-time, frozen-methodology holdout case study |
| **10** | Hardening + release | Full tests, CI/CD audit, final README, honest resume metrics |

Every Integrity Audit gate, stop/gate criterion, and phase acceptance criterion is defined in [`BUILD_PLAN.md`](BUILD_PLAN.md).

---

## Tech stack

| Technology | Use in RiskFecta |
|------------|------------------|
| Python 3.10–3.13 | Core language |
| PyTorch | Pooled LSTM |
| XGBoost | Pooled tabular return model |
| scikit-learn | Preprocessing, evaluation helpers, Ledoit-Wolf shrinkage |
| SciPy | Constrained mean-variance portfolio optimization (SLSQP) |
| PostgreSQL (Supabase-hosted) | Time-series + forecast + portfolio store |
| pandas / NumPy / pandas-ta | Data handling and backward-looking indicators |
| FastAPI | Thin backend API layer |
| React + TypeScript | Frontend |
| Plotly | Interactive charts |
| Bloomberg Terminal | Data source (BQL / Excel → CSV only; no API scripting on laptop) |

## Repository layout

```
RiskFecta/
├── app/                       # FastAPI backend (Phase 2A skeleton: main.py, db.py, schemas.py, routes/)
├── frontend/                  # React + TypeScript + Plotly frontend (Phase 2B: src/api, src/components)
├── pipeline/                  # ingest.py, features.py (Phases 1 & 3)
├── models/                    # baselines.py, xgboost_model.py, lstm.py, ensemble.py (Phases 4-6)
├── optimizer/                 # covariance.py, portfolio.py (Phase 7)
├── tests/                     # pytest suite
├── data/raw/                  # Bloomberg CSVs (gitignored)
├── config.py                  # Universe, features, rolling-window constants
├── schema.sql                 # PostgreSQL DDL
├── requirements.txt
├── docs/
│   ├── SETUP.md
│   └── Bloomberg_export_spec.md
├── PRD.md / TRD.md / ML_SPEC.md / BUILD_PLAN.md   # V2 source of truth
└── PRD_and_buildplan/archive/  # V1 material — historical reference only, does not govern V2
```

---

## Getting started (developers)

1. **Clone** the repository and use **Python 3.10–3.13** (see [docs/SETUP.md](docs/SETUP.md)).
2. Create a virtual environment and install dependencies:
   ```powershell
   py -3.13 -m venv venv
   .\venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   ```
3. **PostgreSQL (Supabase-hosted):** create a Supabase project, apply `schema.sql`, and set `.env` from `.env.example`:
   ```
   DATABASE_URL=postgresql://postgres.PROJECT_REF:PASSWORD@aws-0-REGION.pooler.supabase.com:6543/postgres
   ```
4. **Verify config:**
   ```powershell
   python -c "import config; print(config.TRAIN_WINDOW, config.STEP, config.LSTM_SEQ, config.FORECAST_HORIZON)"
   ```

Full setup steps: **[docs/SETUP.md](docs/SETUP.md)**.
Bloomberg export field list: **[docs/Bloomberg_export_spec.md](docs/Bloomberg_export_spec.md)**.

---

## Running the API locally

With `.env` configured (see step 3 above) and dependencies installed:

```powershell
uvicorn app.main:app --reload
```

Interactive docs (auto-generated from the Pydantic schemas): `http://127.0.0.1:8000/docs`

Read-only endpoints (Phase 2A — no forecasts, portfolios, or risk metrics; those tables are empty):

| Endpoint | Purpose |
|----------|---------|
| `GET /health` | Process liveness only |
| `GET /health/ready` | PostgreSQL connectivity check (no credentials in the response) |
| `GET /api/universe` | The real 50-ticker universe from `prices_raw`, with read-only sector metadata where the local static snapshot is available |
| `GET /api/prices/{ticker}?start=&end=` | Chronological OHLCV + `total_return_idx` history from `prices_raw`; 404 for an unknown ticker, 400 if `start` is after `end` |
| `GET /api/market/summary` | Ticker count, row count, and first/last available date — descriptive facts only |

Environment variables:

| Variable | Required | Purpose |
|----------|----------|---------|
| `DATABASE_URL` | Yes | Supabase/Postgres connection string (unchanged from Phase 1) |
| `CORS_ORIGINS` | No | Comma-separated allowed origins for the future React frontend. Defaults to `http://localhost:3000,http://localhost:5173` (CRA/Vite dev servers). Set explicitly, never wildcarded, outside local development. |

An API versioning prefix (e.g. `/api/v1/...`) is deferred, per [TRD.md](TRD.md) §10 — Phase 2A uses a plain `/api/...` prefix; introducing a version prefix later is a documented decision, not a silent breaking change.

---

## Running the frontend locally

The Phase 2B frontend (`frontend/`) is a Vite + React + TypeScript app that consumes the Phase 2A API — it never queries PostgreSQL directly.

1. Install dependencies:
   ```powershell
   cd frontend
   npm install
   ```
2. (Optional) point it at a non-default backend URL — copy `.env.example` to `.env` and set:
   ```
   VITE_API_BASE_URL=http://127.0.0.1:8000
   ```
   If unset, it defaults to `http://127.0.0.1:8000` (the local `uvicorn` default from the section above).
3. Run the backend (`uvicorn app.main:app --reload`, from the repo root) and the frontend together:
   ```powershell
   npm run dev
   ```
   Open `http://localhost:5173`.

What it shows: RiskFecta branding, a build-status panel honestly marked "live" (historical data) vs. "planned" (forecasts, portfolio optimization), the real market summary and universe from the API, and a ticker detail view with Plotly close-price and volume charts plus date-range filtering — all sourced from `GET /api/*`, nothing fabricated.

Other commands (run from `frontend/`):

| Command | Purpose |
|---------|---------|
| `npm run build` | Type-check (`tsc -b`) + production build to `frontend/dist/` |
| `npm test` | Run the Vitest suite |
| `npm run lint` | Lint with oxlint |

Environment variables:

| Variable | Required | Purpose |
|----------|----------|---------|
| `VITE_API_BASE_URL` | No | Base URL of the FastAPI backend. Not a secret — safe to bake into the client bundle. Defaults to `http://127.0.0.1:8000`. |

Make sure the backend's `CORS_ORIGINS` (see above) includes the frontend's dev origin — `http://localhost:5173` is already in its default.

---

## Data & privacy

- Bloomberg CSVs live under `data/raw/` and are **gitignored**; they are treated as immutable and are never hand-edited.
- Never commit `.env`, credentials, or raw market exports.
- Genuine missing Bloomberg values remain **NULL** in the database; forward-fill happens only in feature engineering (Phase 3), never at ingest, per-ticker, past-only.
- Macro series and static snapshot fields are validated/normalized (Phase 1) but not persisted to PostgreSQL yet — see [BUILD_PLAN.md](BUILD_PLAN.md) Phase 1's Macro / Static Persistence Note. Static fields are never attached to historical `(ticker, date)` rows.

---

## Scope boundaries

RiskFecta intentionally does **not** include:

- Live trading, paper trading, or order execution.
- Personalized investment advice.
- Synthetic or Yahoo Finance substitutes for MVP (real Bloomberg export required).
- Bloomberg API scripting from a personal laptop.
- Leverage, shorting, Black-Litterman, risk parity, or robust optimization (unless separately approved).
- Learned ticker embeddings or ordinal ticker IDs as default MVP model features.
- Consumer budgeting/transaction tracking or systemic/graph-based contagion research — see [PRD.md § Differentiation](PRD.md#5-differentiation).

See [PRD.md § Non-Goals](PRD.md#8-non-goals) for the complete list.

---

## Author & context

**Kapil Iyer** — University of Waterloo
Academic / internship-tier quantitative portfolio research project (2026). Architecture and ML methodology are specified in [`PRD.md`](PRD.md), [`TRD.md`](TRD.md), and [`ML_SPEC.md`](ML_SPEC.md) for reproducibility and technical review.

For historical V1 planning material (superseded, not authoritative), see `PRD_and_buildplan/archive/`.

---

## License

License TBD. Contact the repository owner before reuse or redistribution.
