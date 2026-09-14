# RiskFecta — Technical Requirements Document (V2)

**Status:** Draft V2 — Phase 0 specification package
**Supersedes:** `PRD_and_buildplan/RiskFecta_BuildPlan.docx` (V1, technical portions). See [PRD.md — V1 Archive / Supersession](PRD.md#v1-archive--supersession).
**Authority:** System/technical architecture. Product scope lives in [`PRD.md`](PRD.md); ML/quant methodology lives in [`ML_SPEC.md`](ML_SPEC.md); execution sequencing lives in [`BUILD_PLAN.md`](BUILD_PLAN.md). This document does **not** define model-training methodology — that belongs to ML_SPEC.

---

## 1. System Architecture

```
Bloomberg CSV exports (data/raw/, gitignored, immutable)
        │
        ▼
Validation + normalization  (wide → long, trading-session filtering, NULL-preserving)
        │
        ▼
Supabase-hosted PostgreSQL  (5-table schema — see schema.sql)
        │
        ▼
Python feature / ML / optimization modules  (pipeline/, models/, optimizer/)
        │
        ▼
FastAPI  (thin API layer)
        │
        ▼
React + TypeScript  (frontend)  +  Plotly (chart rendering)
        │
        ▼
Deployed web application
```

## 2. Component Boundaries

| Component | Responsibility | Must NOT do |
|---|---|---|
| `pipeline/` | Bloomberg CSV validation, wide→long normalization, trading-session filtering, feature engineering | Fit scalers/selectors on future data; fabricate values for missing data |
| `models/` | Baselines, pooled XGBoost, pooled LSTM, ensemble | Use future information; own HTTP routes |
| `optimizer/` | Covariance estimation, constrained mean-variance optimization, efficient frontier | Use predicted or future-realized returns for covariance; use future returns anywhere |
| `app/` (FastAPI) | Thin HTTP/API layer over the above modules and the database | Contain ML/optimizer business logic — it calls into `pipeline/`, `models/`, `optimizer/` |
| `frontend/` (React + TypeScript) | Presentation, Plotly chart rendering, user workflows from [PRD.md](PRD.md) §6, §9 | Contain business logic that belongs server-side |
| `schema.sql` | Canonical PostgreSQL DDL | — |
| `config.py` | Universe, rolling-window constants, feature-column lists, paths | Encode leakage-prone predictive features (see §21 and [ML_SPEC.md](ML_SPEC.md) §10) |

ML and optimizer logic stay in ordinary Python modules independent of HTTP routes — FastAPI is a thin service/API layer, never a microservice mesh.

## 3. Repository Structure

Current (Phase 0 scaffold):

```
RiskFecta/
├── app/                  # FastAPI backend (V2) — app/pages/ is a V1 Streamlit remnant, see §21
├── pipeline/             # ingest + feature engineering
├── models/               # baselines, xgboost, lstm, ensemble
├── optimizer/            # covariance + mean-variance optimizer
├── tests/                # pytest suite
├── data/raw/             # Bloomberg CSVs (gitignored, immutable)
├── data/processed/       # derived/feature datasets (gitignored)
├── config.py             # universe, rolling-window constants, feature lists, paths
├── schema.sql            # PostgreSQL DDL (5 tables)
├── requirements.txt
├── .env.example
├── docs/
│   ├── SETUP.md
│   └── Bloomberg_export_spec.md
├── PRD.md / TRD.md / ML_SPEC.md / BUILD_PLAN.md   # V2 source of truth (this package)
└── PRD_and_buildplan/    # V1 archive — historical reference only
```

A `frontend/` directory (React + TypeScript) does not yet exist and is created in Phase 2 (see [BUILD_PLAN.md](BUILD_PLAN.md)).

## 4. Data Flow

1. Bloomberg BQL/Excel exports land in `data/raw/` (gitignored, immutable, never modified in place).
2. A validation + normalization step reshapes wide Bloomberg panel format to long `(ticker, date, field)` rows, filters to valid trading sessions (PX_LAST-gated), and preserves genuine missing values as NULL.
3. Normalized rows load into `prices_raw`, static snapshot fields into a static-fields store, macro series alongside.
4. `pipeline/features.py` computes engineered features per `(ticker, date)` into `features`, respecting per-ticker, past-only forward-fill rules.
5. `models/` consumes `features` + `prices_raw`-derived targets under rolling walk-forward splits, writing forecasts to `predictions`.
6. `optimizer/` consumes ensemble forecasts (expected returns) and historical realized returns (covariance) to produce `portfolios` and `risk_metrics`.
7. FastAPI exposes read endpoints over these tables (and, where relevant, triggers pipeline/model/optimizer runs) to the React frontend.

## 5. PostgreSQL / Supabase Role

- Supabase provides **managed PostgreSQL only**. The application depends on a standard `DATABASE_URL` connection string.
- Do not introduce an unnecessary dependency on Supabase Auth, Storage, Edge Functions, or Realtime unless a specific, approved feature requires it.
- Local development may use a local PostgreSQL instance with an equivalent `DATABASE_URL`; Supabase is the default deployed/production database.

## 6. Schema Usage

The current 5-table schema (`schema.sql`) is the V2 baseline and is compatible with this architecture:

1. `prices_raw` — immutable Bloomberg price/total-return load. `close NOT NULL` (session-gated on PX_LAST at load time); `total_return_idx` nullable. `UNIQUE(ticker, date)`.
2. `features` — engineered technical/macro features per `(ticker, date)`. **Note:** currently also carries `beta`, `mkt_cap_log`, `div_yield`, `sector` columns inherited from V1 — see the static-feature leakage policy in [ML_SPEC.md](ML_SPEC.md) §10 and the known-conflict note in [BUILD_PLAN.md](BUILD_PLAN.md) Phase 0. Storing these columns is not itself prohibited (descriptive/UI use is allowed); using them as **predictive model inputs** without point-in-time justification is prohibited.
3. `predictions` — per-model and ensemble forecasts, `UNIQUE(ticker, forecast_date)`, `actual_return`/`directional_correct` filled post-hoc once realized.
4. `portfolios` — optimizer output per `run_id`/point on the frontier. `run_id` is a logical/application key, not a foreign key.
5. `risk_metrics` — portfolio- and asset-level risk diagnostics per `run_id`.

Any schema change required to fully align with the ML Spec's leakage policy is scoped to Phase 0/Phase 1 of the Build Plan, executed only after this documentation package is approved — not as part of this specification task.

## 7. Configuration / Secrets

- All secrets (`DATABASE_URL`, any future API keys) are read from environment variables via `.env` (gitignored); `.env.example` documents required keys with placeholder values only.
- `config.py` is the single source of truth for the ticker universe, rolling-window constants (`TRAIN_WINDOW`, `STEP`, `LSTM_SEQ`, `FORECAST_HORIZON`), and feature-column lists referenced by the pipeline and models.
- Never hardcode credentials, connection strings, or API keys in source, tests, or documentation.

## 8. Python Package Boundaries

- `pipeline/`, `models/`, `optimizer/` are independent, importable packages with no circular dependencies; `models/` and `optimizer/` depend on `pipeline/` outputs (DB rows / dataframes), not on each other except where the ensemble module reads both `models/xgboost_model.py` and `models/lstm.py` outputs.
- `app/` (FastAPI) imports from `pipeline/`, `models/`, `optimizer/` — never the reverse.
- `config.py` sits at the root and may be imported by any package.

## 9. FastAPI Architecture

- FastAPI is a **thin** layer: request validation (Pydantic schemas), calling into `pipeline/`/`models/`/`optimizer/` or querying the database, and response serialization. No business/statistical logic lives in route handlers.
- Organize routes by resource area matching the PRD's user-facing surfaces (universe/market data, forecasts, model comparison, portfolio construction, risk analytics), each as its own router module.
- No microservices: one FastAPI application, one deployable backend unit.

## 10. API Contract Strategy

- Pydantic models define request/response schemas; OpenAPI/Swagger docs are generated automatically by FastAPI and treated as the live contract reference for the frontend.
- Versioning approach (e.g., `/api/v1/...` prefix) is a Phase 2 implementation decision, not fixed here — flagged as a decision gate at build time, not invented in this document.

## 11. React + TypeScript Structure

- A conventional component-based structure: pages matching the PRD's user-facing surfaces, shared chart components wrapping Plotly, a typed API client generated or hand-written from the FastAPI OpenAPI schema.
- State management approach (React Query / plain fetch+hooks / other) is a Phase 2 implementation decision — not fixed here.

## 12. Plotly Integration

- Charts render client-side via `plotly.js` (React wrapper, e.g. `react-plotly.js` or equivalent) fed by JSON data from FastAPI endpoints.
- No server-side chart image generation; the frontend owns rendering.

## 13. Deployment Architecture

- **Frontend:** likely Vercel (React/TypeScript static + serverless-friendly hosting).
- **Backend:** likely Render (FastAPI).
- **Database:** Supabase-hosted PostgreSQL.
- Exact provider choices remain configurable — if a concrete constraint (cost, Bloomberg data residency, existing account) forces a different provider, that is a TRD amendment, not a silent deviation.
- One public URL persists across phases; each phase's deployment reflects genuinely current repo/pipeline/database state (see [PRD.md](PRD.md) §12 Data Freshness Policy and the Website/Truthfulness rule in [BUILD_PLAN.md](BUILD_PLAN.md)).

## 14. CI/CD Expectations

- GitHub Actions (or equivalent) runs the test suite (`pytest`) and, where applicable, frontend build/lint checks on every push/PR to `main`.
- Deployment to the hosting providers above is triggered from CI on merge to `main` (or a manually gated release step), not from ad hoc local pushes.
- CI must not require Bloomberg-sourced raw data (which is gitignored and machine-local) to run unit tests — tests that need real data use fixtures or a documented local-only marker.

## 15. Environment Separation

- **Local development:** local `.env` with either a local PostgreSQL instance or a dev Supabase project; local Bloomberg CSVs under `data/raw/`.
- **Production/deployed:** production Supabase project, production `DATABASE_URL`, secrets set via the hosting provider's secret manager — never committed.
- No shared mutable state between environments; each environment's database is independently seeded from the same immutable raw exports.

## 16. Observability / Error Handling

- FastAPI: structured error responses (consistent JSON error shape), request logging, and 5xx alerting appropriate to a small deployed service (exact tool — e.g. Sentry — is a Phase 2 decision gate, not fixed here).
- Pipeline/model/optimizer runs log enough to reconstruct which formation date, window, and code version produced a given `predictions`/`portfolios` row.

## 17. Testing Strategy

- `pytest` covers: wide→long normalization correctness, trading-session filtering, target/feature leakage checks (see [ML_SPEC.md](ML_SPEC.md) §29 Leakage Checklist), rolling-window split logic, ensemble arithmetic, optimizer constraint satisfaction, and API contract smoke tests.
- Tests run against fixture data, not the real gitignored Bloomberg export, so CI can run without local data.
- No implementation-level test detail is fixed here beyond this strategy — concrete test files are a Build Plan deliverable per phase.

## 18. Security Basics

- No secrets in source control; `.env` and `data/raw/` stay gitignored.
- Database access via least-privilege credentials appropriate to Supabase's connection pooling model.
- No public write endpoints without appropriate validation; the MVP has no authenticated user accounts unless a future feature requires them (not currently planned).
- Dependencies pinned in `requirements.txt` / frontend lockfile; no arbitrary code execution from user input.

## 19. Data Immutability Policy

- Files under `data/raw/` are never modified in place, never overwritten silently, and never hand-edited. Corrections happen only by re-exporting from Bloomberg and adding a new file, with the change documented.
- `prices_raw` (once loaded) is treated as immutable historical record; corrections are additive/documented, not silent updates.

## 20. Production / Demo State Policy

Every deployment, at every phase, reflects genuinely current repository, pipeline, and database state:

- No fabricated model predictions, placeholder metrics presented as real, fake backtest results, or fake portfolio recommendations at any phase — including the earliest Phase 2 deployment (see [BUILD_PLAN.md](BUILD_PLAN.md) Phase 2 and the Website/Truthfulness rule).
- Unfinished research features are clearly labeled as unfinished, not hidden or faked.

## 21. Compatibility With Existing Five-Table Schema

The current `schema.sql` (5 tables: `prices_raw`, `features`, `predictions`, `portfolios`, `risk_metrics`) is compatible with the V2 architecture with two known, documented conflicts to resolve during Phase 0/1 implementation (not in this documentation task):

1. `features` carries `beta`, `mkt_cap_log`, `div_yield`, `sector` columns that, per [ML_SPEC.md](ML_SPEC.md) §10, must **not** be used as predictive model inputs without point-in-time justification. Storing/displaying them is fine; `config.py`'s `XGBOOST_FEATURE_COLS` currently includes `beta`, `mkt_cap_log`, `div_yield`, `sector` as if they were valid predictive inputs — flagged as a known repo conflict, to be corrected only in Phase 0 after this documentation package is approved.
2. `app/pages/` is a V1 Streamlit-era directory name; it is repurposed or removed when the FastAPI + React architecture is scaffolded in Phase 2.

## 22. Provider Portability

- The architecture depends on standard PostgreSQL (via `DATABASE_URL`), standard HTTP (FastAPI), and standard static/serverless frontend hosting — no proprietary Supabase-only or Vercel-only/Render-only feature is required, so switching managed-Postgres, frontend-hosting, or backend-hosting providers should not require architectural changes, only redeployment.
