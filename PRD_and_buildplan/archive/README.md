# V1 Archive — Historical Reference Only

Everything in this folder is **RiskFecta V1** material: the original PRD, build plan, and Phase 0 plan (as `.docx` originals and plain-text extractions), moved here unmodified.

**This folder does not govern RiskFecta V2.** The current source of truth is, in order of authority: [`../../PRD.md`](../../PRD.md), [`../../TRD.md`](../../TRD.md), [`../../ML_SPEC.md`](../../ML_SPEC.md), [`../../BUILD_PLAN.md`](../../BUILD_PLAN.md).

Known V1 decisions superseded in V2 (see `PRD.md` — V1 Archive / Supersession for the full list):

- 30-day/30-session forecast horizon → V2 uses **21 trading sessions**.
- Streamlit + Tableau → V2 uses **FastAPI + React/TypeScript + Plotly**; no Streamlit, no Tableau.
- Localhost-first PostgreSQL → V2 defaults to **Supabase-hosted PostgreSQL**.
- Unrestricted predictive use of snapshot static fields (beta, market cap, dividend yield) → V2 **excludes** these from predictive training without point-in-time justification.
- Placeholder/loosely-specified ticker universe → V2 uses the **exact validated 50-symbol Bloomberg export universe** (including MRSH, not MMC).

Nothing here was deleted or rewritten — only moved, so historical context stays available without controlling new architecture decisions.
