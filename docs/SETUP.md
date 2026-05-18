# RiskFecta — Environment setup

## Python

Use **Python 3.10–3.13** (pandas-ta does not support 3.14). You have 3.13; use it.

## Commands (PowerShell, run from project root)

```powershell
cd c:\Users\kapil\RiskFecta

# Create venv with Python 3.13 (py launcher)
py -3.13 -m venv venv

# Activate venv
.\venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

If you don’t have the `py` launcher, use the full path to Python 3.13, e.g.:

```powershell
& "C:\Path\To\Python313\python.exe" -m venv venv
```

## PostgreSQL (Windows)

If `createdb` or `psql` are not recognized, either:

**Option A — Add PostgreSQL to PATH**

1. Find the PostgreSQL `bin` folder (default for v18: `C:\Program Files\PostgreSQL\18\bin`).
2. Add it to your user PATH, or run in PowerShell (use 18 if that’s your version):

```powershell
$env:Path += ";C:\Program Files\PostgreSQL\18\bin"
createdb -U postgres riskfecta
psql -U postgres -d riskfecta -f schema.sql
psql -U postgres -d riskfecta -c "\dt"
```

**Option B — Use full path to psql**

```powershell
& "C:\Program Files\PostgreSQL\18\bin\psql.exe" -U postgres -c "CREATE DATABASE riskfecta;"
& "C:\Program Files\PostgreSQL\18\bin\psql.exe" -U postgres -d riskfecta -f schema.sql
```

**Option C — pgAdmin**

Create database `riskfecta`, then open Query Tool and run the contents of `schema.sql`.

---

## .env file

Create a file named `.env` in the project root (same folder as `config.py`) with:

```
DATABASE_URL=postgresql://postgres:YOUR_PASSWORD@localhost:5432/riskfecta
```

Replace `YOUR_PASSWORD` with your PostgreSQL password. The app loads `.env` via python-dotenv; do not commit `.env`.

**Test (with venv active):**

```powershell
python -c "from config import get_database_url; print(get_database_url())"
```
