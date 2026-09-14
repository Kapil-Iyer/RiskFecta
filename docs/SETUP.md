# RiskFecta — Environment setup (V2 baseline)

Full V2 deployment documentation will be rewritten with the specification package.
This file covers local Python setup and how the app expects PostgreSQL.

## Python

Use **Python 3.10–3.13** (pandas-ta does not support 3.14).

```powershell
cd c:\Users\kapil\RiskFecta

py -3.13 -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Database (PostgreSQL via Supabase)

RiskFecta V2 uses **PostgreSQL hosted through Supabase** (managed Postgres).
Application code should use a standard `DATABASE_URL` — not Supabase Auth/Storage APIs.

1. Create a Supabase project and obtain the Postgres connection URI.
2. Apply `schema.sql` to that database (SQL editor or `psql` against the URI).
3. Copy `.env.example` → `.env` and set `DATABASE_URL` to your URI (never commit `.env`).

A locally installed PostgreSQL server is **not** the V2 default architecture.

**Test (venv active):**

```powershell
python -c "from config import get_database_url; print(get_database_url())"
```

## Note

Raw Bloomberg CSVs live under `data/raw/` (gitignored). Do not modify them.
