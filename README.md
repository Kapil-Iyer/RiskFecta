# RiskFecta

**Quantitative Portfolio Intelligence Platform**

RiskFecta is a Python-based portfolio analytics system that combines machine-learning return forecasting with Modern Portfolio Theory optimization. It ingests institutional-grade market data from Bloomberg Terminal, trains a hybrid **LSTM + XGBoost** ensemble to predict 30-day forward returns, and feeds those signals into a **SciPy Efficient Frontier** optimizer—then surfaces results through an interactive **Streamlit** demo and a professional **Tableau** report.

> **Status (May 2026):** Early development — **Phase 0 (pre-build)** complete. Repository scaffold, database schema, configuration, and documentation are in place. Data pipeline, models, optimizer, and deployment are planned per the locked PRD and build plan (Phases 1–9).

---

## The problem we are solving

Professional portfolio tools (e.g. Bloomberg PORT, FactSet) cost tens of thousands of dollars per year and are out of reach for most retail investors. Free alternatives typically rely on historical averages for expected returns and do not combine deep learning, factor models, and rigorous optimization in one reproducible, deployable stack.

**RiskFecta’s goal:** Build an accessible, interview-defensible quant platform that:

1. Sources real Bloomberg data (BQL / Excel export → CSV → PostgreSQL)
2. Forecasts 30-day returns with a **dual-model ensemble** (sequential LSTM + tabular XGBoost)
3. Optimizes long-only portfolios on the **Efficient Frontier** using ensemble expected returns and realized-return covariance
4. Presents outcomes interactively (Streamlit + Plotly) and in a recruiter-facing static report (Tableau)

RiskFecta is **not** a live trading or order-execution system. It is a research and analytics platform with strict out-of-sample validation and look-ahead controls.

---

## Architecture (target state)

```mermaid
flowchart LR
    subgraph ingest [Phase 1–2]
        BB[Bloomberg BQL / Excel]
        CSV[CSV exports]
        PG[(PostgreSQL)]
        BB --> CSV --> PG
    end

    subgraph ml [Phase 3–5]
        LSTM[LSTM — price sequences]
        XGB[XGBoost — factors / macro]
        ENS[50/50 ensemble]
        LSTM --> ENS
        XGB --> ENS
    end

    subgraph opt [Phase 6]
        MPT[SciPy MPT optimizer]
        EF[Efficient Frontier]
        ENS --> MPT --> EF
    end

    subgraph viz [Phase 7–8]
        ST[Streamlit Cloud]
        TB[Tableau report]
        EF --> ST
        EF --> TB
    end

    PG --> LSTM
    PG --> XGB
    PG --> MPT
```

| Layer | Role |
|--------|------|
| **Bloomberg Terminal** | Raw OHLCV, total return, static equity fields, macro (VIX, 10Y yield, SPX) |
| **Python pipeline** | Ingestion, backward-looking technical indicators (RSI, MACD, Bollinger, momentum, vol) |
| **LSTM (PyTorch)** | 60-day sequences → 30-day return forecast |
| **XGBoost** | Tabular factors (beta, cap, sector, macro, momentum) → 30-day return forecast |
| **Ensemble** | `0.5 × LSTM + 0.5 × XGBoost` → expected return vector |
| **Optimizer (SciPy)** | Long-only Efficient Frontier; covariance from **realized** returns only |
| **Streamlit + Plotly** | Live multi-page demo (weights, frontier, predictions, model metrics, risk) |
| **Tableau** | Three static dashboards for portfolio, model, and risk analytics |

**Validation (locked):** Rolling window — **252** trading-day train, **21**-day step, **30**-day forecast horizon; final 3 months held out for final evaluation. No expanding window; no regime classifier. Risk-free rate for Sharpe: **USGG10YR / 252** everywhere.

---

## Current state vs. roadmap

### What exists today (Phase 0)

| Deliverable | Status |
|-------------|--------|
| Repo layout (`pipeline/`, `models/`, `optimizer/`, `app/`, `tests/`, `tableau/`) | Done |
| `requirements.txt`, `config.py` (universe, features, rolling constants) | Done |
| `schema.sql` (5 tables: prices, features, predictions, portfolios, risk_metrics) | Done |
| `docs/SETUP.md`, `docs/Bloomberg_export_spec.md` | Done |
| Local PostgreSQL + `.env` | **Your machine** — see [docs/SETUP.md](docs/SETUP.md) |
| Bloomberg CSV data pull| Done |
| Bloomberg CSV ingestion | Planned|
| Feature engineering, ML, optimizer, UI | Planned — Phases 2–8 |

**MVP-A** (minimum viable product) is complete when: real Bloomberg data is ingested; rolling-window LSTM and XGBoost run without leakage; ensemble and Efficient Frontier are computed; Streamlit Cloud demo runs on that data. See `PRD_and_buildplan/Phase0_Plan.md` for criteria.

### Build phases (projection)

Aligned with the locked [Build Plan](PRD_and_buildplan/BuildPlan_extracted.txt) (~85–90 hrs total):

| Phase | Focus | Key output |
|-------|--------|------------|
| **0 — Pre-build** | Environment, schema, config | Repo + DB scaffold *(current)* |
| **1** | Bloomberg pull + `ingest.py` | `prices_raw` populated; UNIQUE constraints verified |
| **2** | `features.py` | Technical + factor features in `features` |
| **3** | `lstm.py` (+ leakage audit) | `lstm_pred` in `predictions` |
| **4** | `xgboost_model.py` | `xgb_pred` in `predictions` |
| **5** | `ensemble.py` | `ensemble_pred`, directional accuracy, evaluation |
| **6** | `portfolio.py` | Efficient Frontier → `portfolios`, `risk_metrics` |
| **7** | Streamlit app + cloud DB | **Live demo URL** (5 pages, Plotly) |
| **8** | Tableau | **Public link or PDF** — 3 dashboards |
| **9** | Tests, polish, portfolio embed | `pytest`, README links, release tag |

Live links will be added here after Phase 7–8:

- **Streamlit demo:** *coming soon*
- **Tableau report:** *coming soon*

---

## Tech stack

| Technology | Use in RiskFecta |
|------------|------------------|
| Python 3.10–3.13 | Core language |
| PyTorch | LSTM |
| XGBoost | Tabular return model |
| scikit-learn | Preprocessing, evaluation helpers |
| SciPy | Portfolio optimization (SLSQP) |
| PostgreSQL | Time-series store (local dev; Supabase/Neon for deploy) |
| pandas / NumPy / pandas-ta | Data + backward-looking indicators |
| Plotly + Streamlit | Interactive web app |
| Tableau | Static professional report |
| Bloomberg Terminal | Data source (BQL / Excel → CSV only; no API scripting on laptop) |

---

## Repository layout

```
RiskFecta/
├── app/                 # Streamlit app (Phase 7)
├── pipeline/            # ingest.py, features.py (Phases 1–2)
├── models/              # lstm.py, xgboost_model.py, ensemble.py (Phases 3–5)
├── optimizer/           # portfolio.py (Phase 6)
├── tests/               # Pipeline & model tests (Phase 9)
├── tableau/             # Tableau workbook / exports (Phase 8)
├── data/raw/            # Bloomberg CSVs (gitignored)
├── config.py            # Universe, features, rolling-window constants
├── schema.sql           # PostgreSQL DDL
├── requirements.txt
├── docs/
│   ├── SETUP.md
│   └── Bloomberg_export_spec.md
└── PRD_and_buildplan/   # Locked PRD v1.0 + build plan (reference)
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
3. **PostgreSQL:** Create database `riskfecta`, apply `schema.sql`, add `.env` from `.env.example`:
   ```
   DATABASE_URL=postgresql://user:password@localhost:5432/riskfecta
   ```
4. **Verify config:**
   ```powershell
   python -c "import config; print(config.TRAIN_WINDOW, config.STEP)"
   ```

Full Windows/PostgreSQL steps: **[docs/SETUP.md](docs/SETUP.md)**.  
Bloomberg export field list for Phase 1: **[docs/Bloomberg_export_spec.md](docs/Bloomberg_export_spec.md)**.

---

## Data & privacy

- Bloomberg CSVs live under `data/raw/` and are **gitignored**.
- Never commit `.env`, credentials, or raw market exports.
- Missing Bloomberg values remain **NULL** in the database; forward-fill happens only in feature engineering (Phase 2), not at ingest.

---

## Scope boundaries

RiskFecta intentionally does **not** include:

- Live trading or paper trading
- Synthetic or Yahoo Finance substitutes for MVP (real Bloomberg export required)
- Bloomberg API scripting from a personal laptop
- SQLite / Power BI (those belong to a separate project, PlainCents)

---

## Author & context

**Kapil Iyer** — University of Waterloo  
Academic / internship-tier quant portfolio project (2026). Architecture and ML decisions are pre-validated in the PRD for reproducibility and technical interviews.

For internal planning detail, see `PRD_and_buildplan/` (PRD v1.0, build plan v1.0, Phase 0 checklist).

---

## License

License TBD. Contact the repository owner before reuse or redistribution.
