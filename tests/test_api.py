"""
Phase 2A CI-safe unit tests for the FastAPI skeleton (app/). These never
require a live database — the `get_db` dependency is overridden with an
in-memory fake connection/cursor via FastAPI's dependency_overrides. Real
database behavior (against the live Supabase Postgres) is covered separately
in tests/test_api_db_integration.py, which follows the existing `db` marker
pattern (tests/test_db_integration.py) and is skipped when DATABASE_URL is
unset (e.g. in CI).
"""
from __future__ import annotations

import ast
import pathlib
import re
from datetime import date

import numpy as np
import pandas as pd
import psycopg2
import pytest
from fastapi.testclient import TestClient

import config
from app.db import get_db
from app.main import app
from app.routes import backtest as backtest_module
from app.routes import frontier as frontier_module
from app.routes import risk as risk_module

_SQL_MUTATION_KEYWORDS = re.compile(r"\b(insert\s+into|update\s+\w+\s+set|delete\s+from|drop\s+table|truncate|alter\s+table)\b")


class _FakeCursor:
    def __init__(self, rows):
        self._rows = rows

    def execute(self, sql, params=None):
        pass

    def fetchall(self):
        return self._rows

    def fetchone(self):
        return self._rows[0] if self._rows else None

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False


class _FakeConnection:
    def __init__(self, rows):
        self._rows = rows

    def cursor(self):
        return _FakeCursor(self._rows)

    def close(self):
        pass


PRICE_ROWS = [
    (date(2024, 1, 2), 100.0, 101.0, 99.0, 100.5, 1_000_000, 250.0),
    (date(2024, 1, 3), 100.5, 102.0, 100.0, 101.5, 1_100_000, 252.0),
    (date(2024, 1, 4), 101.5, 103.0, 101.0, 102.5, 1_050_000, 254.0),
]


def _get_db_returning(rows):
    def _fake_get_db():
        yield _FakeConnection(rows)

    return _fake_get_db


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------
def test_health_ok(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_readiness_reports_unavailable_when_db_unreachable(client, monkeypatch):
    import app.routes.health as health_module

    monkeypatch.setattr(health_module, "check_connectivity", lambda: False)
    resp = client.get("/health/ready")
    assert resp.status_code == 503
    assert resp.json() == {"status": "error", "database": "unavailable"}


def test_readiness_reports_ok_when_db_reachable(client, monkeypatch):
    import app.routes.health as health_module

    monkeypatch.setattr(health_module, "check_connectivity", lambda: True)
    resp = client.get("/health/ready")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok", "database": "connected"}


# ---------------------------------------------------------------------------
# Universe
# ---------------------------------------------------------------------------
def test_universe_returns_config_universe(client):
    tickers = [(t,) for t in sorted(config.TICKER_UNIVERSE)]
    app.dependency_overrides[get_db] = _get_db_returning(tickers)
    resp = client.get("/api/universe")
    assert resp.status_code == 200
    body = resp.json()
    assert {row["ticker"] for row in body} == set(config.TICKER_UNIVERSE)
    assert all("sector" in row for row in body)  # present (possibly null)


# ---------------------------------------------------------------------------
# Prices
# ---------------------------------------------------------------------------
def test_prices_valid_ticker_returns_chronological_rows(client):
    app.dependency_overrides[get_db] = _get_db_returning(PRICE_ROWS)
    resp = client.get("/api/prices/AAPL")
    assert resp.status_code == 200
    body = resp.json()
    assert body["ticker"] == "AAPL"
    assert body["count"] == len(PRICE_ROWS)
    dates = [row["date"] for row in body["prices"]]
    assert dates == sorted(dates)


def test_prices_case_insensitive_uses_canonical_symbol(client):
    app.dependency_overrides[get_db] = _get_db_returning(PRICE_ROWS)
    resp = client.get("/api/prices/aapl")
    assert resp.status_code == 200
    assert resp.json()["ticker"] == "AAPL"


def test_prices_unknown_ticker_returns_404(client):
    app.dependency_overrides[get_db] = _get_db_returning([])
    resp = client.get("/api/prices/NOT_A_REAL_TICKER")
    assert resp.status_code == 404


def test_prices_invalid_date_range_returns_400(client):
    app.dependency_overrides[get_db] = _get_db_returning(PRICE_ROWS)
    resp = client.get("/api/prices/AAPL", params={"start": "2024-02-01", "end": "2024-01-01"})
    assert resp.status_code == 400


def test_prices_date_filtering_reflected_in_response(client):
    app.dependency_overrides[get_db] = _get_db_returning(PRICE_ROWS[:1])
    resp = client.get("/api/prices/AAPL", params={"start": "2024-01-02", "end": "2024-01-02"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["count"] == 1
    assert body["start"] == "2024-01-02"
    assert body["end"] == "2024-01-02"


def test_prices_includes_required_fields(client):
    app.dependency_overrides[get_db] = _get_db_returning(PRICE_ROWS[:1])
    resp = client.get("/api/prices/AAPL")
    row = resp.json()["prices"][0]
    for field in ("date", "open", "high", "low", "close", "volume", "total_return_idx"):
        assert field in row


# ---------------------------------------------------------------------------
# Market summary
# ---------------------------------------------------------------------------
def test_market_summary_shape(client):
    app.dependency_overrides[get_db] = _get_db_returning(
        [(50, 62800, date(2021, 3, 1), date(2026, 2, 27))]
    )
    resp = client.get("/api/market/summary")
    assert resp.status_code == 200
    body = resp.json()
    assert body["ticker_count"] == 50
    assert body["price_row_count"] == 62800
    assert body["first_date"] == "2021-03-01"
    assert body["last_date"] == "2026-02-27"


# ---------------------------------------------------------------------------
# Credential / error-handling safety
# ---------------------------------------------------------------------------
def test_no_credential_leakage_on_db_failure(client):
    """A DB connection failure must surface as a generic 503 — never the raw
    psycopg2 message, which can carry host/user/connection details."""

    def _broken_get_db():
        raise psycopg2.OperationalError(
            'connection to server at "db.supabase.internal-secret-host" (10.0.0.1), '
            "port 5432 failed: FATAL: password authentication failed for user \"postgres\""
        )
        yield  # pragma: no cover - unreachable, keeps this a generator function

    app.dependency_overrides[get_db] = _broken_get_db
    resp = client.get("/api/market/summary")
    assert resp.status_code == 503
    assert resp.json() == {"detail": "Database unavailable"}
    assert "secret-host" not in resp.text
    assert "password" not in resp.text.lower()
    assert "postgres" not in resp.text.lower()


# ---------------------------------------------------------------------------
# Predictions (Phase 4-6 frozen walk-forward forecasts — Phase 8B)
# ---------------------------------------------------------------------------
PREDICTION_ROWS = [
    ("AAPL", date(2022, 3, 25), 0.03, 0.05, 0.04, 0.06, True),
    ("MSFT", date(2022, 3, 25), -0.01, 0.02, 0.005, -0.02, False),
    ("NVDA", date(2022, 3, 25), 0.08, 0.07, 0.075, 0.09, True),
]


def test_prediction_dates_returns_list_of_dates(client):
    rows = [(date(2022, 2, 25),), (date(2022, 3, 28),), (date(2026, 1, 2),)]
    app.dependency_overrides[get_db] = _get_db_returning(rows)
    resp = client.get("/api/predictions/dates")
    assert resp.status_code == 200
    assert resp.json() == ["2022-02-25", "2022-03-28", "2026-01-02"]


def test_predictions_explicit_formation_date_returns_all_model_fields(client):
    app.dependency_overrides[get_db] = _get_db_returning(PREDICTION_ROWS)
    resp = client.get("/api/predictions", params={"formation_date": "2022-02-25"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["formation_date"] == "2022-02-25"
    assert body["target_date"] == "2022-03-25"
    assert body["count"] == 3
    for row in body["predictions"]:
        for field in ("ticker", "xgb_pred", "lstm_pred", "ensemble_pred", "actual_return", "directional_correct"):
            assert field in row


def test_predictions_unknown_formation_date_returns_404(client):
    app.dependency_overrides[get_db] = _get_db_returning([])
    resp = client.get("/api/predictions", params={"formation_date": "2099-01-01"})
    assert resp.status_code == 404


def test_predictions_malformed_formation_date_returns_422(client):
    resp = client.get("/api/predictions", params={"formation_date": "not-a-date"})
    assert resp.status_code == 422


class _FakeCursorDefaultDate:
    """Distinguishes the MAX(forecast_date) lookup from the cross-section
    SELECT within one request — the shared `_FakeCursor` above returns the
    same fixed rows for every call, which can't model this two-query path."""

    def __init__(self, max_date, cross_section_rows):
        self._max_date = max_date
        self._cross_section_rows = cross_section_rows

    def execute(self, sql, params=None):
        pass

    def fetchone(self):
        return (self._max_date,)

    def fetchall(self):
        return self._cross_section_rows

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False


class _FakeConnDefaultDate:
    def __init__(self, max_date, cross_section_rows):
        self._cursor = _FakeCursorDefaultDate(max_date, cross_section_rows)

    def cursor(self):
        return self._cursor

    def close(self):
        pass


def test_predictions_defaults_to_latest_formation_date_when_omitted(client):
    def _fake_get_db():
        yield _FakeConnDefaultDate(date(2026, 1, 2), PREDICTION_ROWS)

    app.dependency_overrides[get_db] = _fake_get_db
    resp = client.get("/api/predictions")
    assert resp.status_code == 200
    assert resp.json()["formation_date"] == "2026-01-02"


# Note: deterministic base ordering (`ORDER BY ticker ASC`) is a real-SQL
# guarantee that a mocked cursor can't meaningfully verify (it returns
# whatever rows it's given, regardless of the query text) — it's covered by
# the live-DB test in test_api_db_integration.py, and separately by a pure
# unit test of the frontend's client-side ranking function
# (frontend/src/pages/forecastRanking.test.ts).


# ---------------------------------------------------------------------------
# Model comparison (Phase 4-6 frozen historical evaluation — Phase 8B)
# ---------------------------------------------------------------------------
def _synthetic_aligned_predictions() -> pd.DataFrame:
    """A tiny, hand-computable stand-in for `load_aligned_predictions`'s
    real (2,350-row) output — this test exercises the ROUTE's wiring
    (provenance labels, metadata, schema), not metric arithmetic, which is
    already covered by tests/test_metrics.py and tests/test_ensemble.py."""
    return pd.DataFrame(
        {
            "ticker": ["AAPL", "MSFT", "AAPL", "MSFT"],
            "forecast_date": pd.to_datetime(["2022-02-25", "2022-02-25", "2022-03-28", "2022-03-28"]),
            "target_date": pd.to_datetime(["2022-03-25", "2022-03-25", "2022-04-27", "2022-04-27"]),
            "xgb_pred": [0.01, -0.02, 0.03, 0.00],
            "lstm_pred": [0.02, -0.01, 0.01, 0.02],
            "actual_return": [0.015, -0.01, 0.02, -0.01],
            "directional_correct": [True, True, True, False],
        }
    )


def test_model_comparison_shape_and_provenance(client, monkeypatch):
    import app.routes.models as models_route

    monkeypatch.setattr(models_route, "load_aligned_predictions", lambda conn: _synthetic_aligned_predictions())
    app.dependency_overrides[get_db] = _get_db_returning([])  # unused by the monkeypatched loader

    resp = client.get("/api/models/comparison")
    assert resp.status_code == 200
    body = resp.json()

    assert body["experiment_type"] == "historical_walk_forward_oos"
    assert body["target_horizon_sessions"] == 21
    assert body["fold_count"] == 2
    assert body["prediction_count"] == 4
    assert body["formation_date_start"] == "2022-02-25"
    assert body["formation_date_end"] == "2022-03-28"

    models_by_key = {m["model"]: m for m in body["models"]}
    assert set(models_by_key) == {"historical_mean", "momentum_3m", "ridge", "xgb_pred", "lstm_pred", "ensemble_pred"}
    for key in ("historical_mean", "momentum_3m", "ridge"):
        assert models_by_key[key]["source"] == "frozen_baseline_constant"
    for key in ("xgb_pred", "lstm_pred", "ensemble_pred"):
        assert models_by_key[key]["source"] == "computed_from_persisted_predictions"

    # Frozen baseline values are the exact authoritative constants, never derived.
    assert models_by_key["historical_mean"]["mae"] == 0.07312
    assert models_by_key["momentum_3m"]["directional_accuracy"] == 0.5077
    assert models_by_key["ridge"]["spearman_corr"] == -0.03153

    metric_keys = {m["key"] for m in body["metric_definitions"]}
    assert metric_keys == {"mae", "rmse", "directional_accuracy", "pearson_corr", "spearman_corr"}
    directions = {m["key"]: m["direction"] for m in body["metric_definitions"]}
    assert directions["mae"] == "lower_is_better"
    assert directions["rmse"] == "lower_is_better"
    assert directions["directional_accuracy"] == "higher_is_better"

    assert body["disagreement"]["source"] == "computed_from_persisted_predictions"
    assert body["disagreement"]["n_total"] == 4


def test_model_comparison_no_overall_winner_field():
    """The response must never carry a single ranked/"best model" field —
    the evidence is metric-mixed by design (§5)."""
    import app.schemas as schemas

    field_names = set(schemas.ModelComparisonResponse.model_fields.keys())
    assert not field_names & {"best_model", "winner", "overall_score", "overall_rank"}


# ---------------------------------------------------------------------------
# Portfolios (frozen, official Phase 7 experiment — Phase 8C)
# ---------------------------------------------------------------------------
class _FakeCursorSequential:
    """Returns a different canned `fetchall()` result on each successive
    `with conn.cursor() as cur:` block — `get_portfolio` issues two
    structurally different SELECTs (portfolio weights, then risk metrics)
    that the shared single-shape `_FakeCursor` can't model."""

    def __init__(self, results):
        self._results = list(results)
        self._call = -1

    def execute(self, sql, params=None):
        pass

    def fetchall(self):
        return self._results[self._call]

    def __enter__(self):
        self._call += 1
        return self

    def __exit__(self, *exc_info):
        return False


class _FakeConnSequential:
    def __init__(self, results):
        self._cursor = _FakeCursorSequential(results)

    def cursor(self):
        return self._cursor

    def close(self):
        pass


def _get_db_returning_sequential(results):
    def _fake_get_db():
        yield _FakeConnSequential(results)

    return _fake_get_db


# 50 rows in config.TICKER_UNIVERSE order, matching real persisted shape —
# two names at the 10% cap, the rest a small non-trivial spread.
_PORTFOLIO_ROWS = [
    (ticker, 0.1 if i < 2 else round(0.9 / 48, 6), 0.041, 0.056, 0.667)
    for i, ticker in enumerate(config.TICKER_UNIVERSE)
]
_RISK_METRIC_ROWS = [("realized_return_21", -0.0123), ("max_weight_observed", 0.1), ("turnover", 0.42)]
_RISK_METRIC_ROWS_FIRST_DATE = [("realized_return_21", -0.0456), ("max_weight_observed", 0.1)]


def test_portfolio_dates_derived_from_official_run_ids(client):
    run_ids = [(f"p7bv1_EQUAL_WEIGHT_2022-03-{d:02d}",) for d in (28, 29, 30)]
    app.dependency_overrides[get_db] = _get_db_returning(run_ids)
    resp = client.get("/api/portfolios/dates")
    assert resp.status_code == 200
    assert resp.json() == ["2022-03-28", "2022-03-29", "2022-03-30"]


def test_portfolio_strategies_returns_five_official_strategies_no_ranking(client):
    resp = client.get("/api/portfolios/strategies")
    assert resp.status_code == 200
    body = resp.json()
    assert [s["key"] for s in body] == [
        "SAMPLE_MINVOL", "SAMPLE_MAXSHARPE", "LW_MINVOL", "LW_MAXSHARPE", "EQUAL_WEIGHT",
    ]
    equal_weight = next(s for s in body if s["key"] == "EQUAL_WEIGHT")
    assert equal_weight["is_optimized"] is False
    assert equal_weight["covariance_estimator"] == "Not applicable"


def test_portfolio_explicit_date_and_strategy_returns_50_holdings_with_provenance(client):
    app.dependency_overrides[get_db] = _get_db_returning_sequential([_PORTFOLIO_ROWS, _RISK_METRIC_ROWS])
    resp = client.get("/api/portfolios", params={"formation_date": "2026-01-02", "strategy": "LW_MAXSHARPE"})
    assert resp.status_code == 200
    body = resp.json()
    assert len(body["holdings"]) == 50
    assert body["strategy"]["key"] == "LW_MAXSHARPE"
    assert body["max_weight_constraint"] == 0.10
    assert body["largest_weight"] == 0.1
    assert body["active_holdings_count"] == 50
    # Construction (ex-ante) figures come straight from the persisted row —
    # never recomputed here.
    assert body["construction"]["expected_return_21"] == 0.041
    assert body["construction"]["predicted_volatility_21"] == 0.056
    assert body["construction"]["expected_sharpe_21"] == 0.667
    # Evaluation (ex-post) figures are clearly separate.
    assert body["evaluation"]["realized_return_21"] == -0.0123
    assert body["evaluation"]["turnover"] == 0.42
    assert body["source"] == "official_phase7_persisted_experiment"


def test_portfolio_first_formation_has_undefined_turnover(client):
    app.dependency_overrides[get_db] = _get_db_returning_sequential([_PORTFOLIO_ROWS, _RISK_METRIC_ROWS_FIRST_DATE])
    resp = client.get("/api/portfolios", params={"formation_date": "2022-03-28", "strategy": "LW_MAXSHARPE"})
    assert resp.status_code == 200
    assert resp.json()["evaluation"]["turnover"] is None  # never 0


def test_equal_weight_has_no_construction_metrics_or_max_weight_constraint(client):
    equal_weight_rows = [(t, 0.02, None, None, None) for t in config.TICKER_UNIVERSE]
    app.dependency_overrides[get_db] = _get_db_returning_sequential([equal_weight_rows, _RISK_METRIC_ROWS])
    resp = client.get("/api/portfolios", params={"formation_date": "2022-03-28", "strategy": "EQUAL_WEIGHT"})
    assert resp.status_code == 200
    body = resp.json()
    assert all(h["weight"] == 0.02 for h in body["holdings"])
    assert body["construction"] == {"expected_return_21": None, "predicted_volatility_21": None, "expected_sharpe_21": None}
    assert body["max_weight_constraint"] is None


def test_portfolio_missing_run_returns_404(client):
    app.dependency_overrides[get_db] = _get_db_returning([])
    resp = client.get("/api/portfolios", params={"formation_date": "2022-02-25", "strategy": "LW_MAXSHARPE"})
    assert resp.status_code == 404


def test_portfolio_invalid_strategy_returns_400(client):
    resp = client.get("/api/portfolios", params={"formation_date": "2022-03-28", "strategy": "NOT_A_STRATEGY"})
    assert resp.status_code == 400


def test_portfolio_malformed_formation_date_returns_422(client):
    resp = client.get("/api/portfolios", params={"formation_date": "not-a-date"})
    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# Efficient Frontier (reconstructed, never persisted — Phase 8C final slice).
# The numeric reconstruction path itself (mu/Sigma/rf, reconciliation
# against official markers, causality) is verified against the REAL
# database in tests/test_api_db_integration.py — these are pure route-level
# unit tests: validation, wiring, response shape, and provenance. Boundary
# functions (`_official_formation_dates`, `reconstruct_mu_sigma_rf`,
# `_official_marker`) are monkeypatched with small synthetic (but
# real-shaped, real-dimension) data so tests stay CI-fast without
# depending on a live database.
# ---------------------------------------------------------------------------
def _fake_official_marker(conn, formation_date, strategy, label):
    from app.schemas import FrontierMarker

    return FrontierMarker(
        label=label, expected_return_21=0.04, volatility_21=0.05, sharpe_21=0.6,
        provenance="official_phase7_persisted",
    )


def _synthetic_frontier_context():
    """A real-dimension (50-asset), deterministic, PSD (mu, Sigma, rf) —
    never derived from or resembling real Phase 7 data (this file must stay
    DB-free)."""
    n = len(config.TICKER_UNIVERSE)
    mu_arr = np.linspace(-0.01, 0.05, n)
    sigma_arr = np.eye(n) * 0.0004  # diagonal, PSD by construction
    rf_21 = 0.002
    return mu_arr, sigma_arr, rf_21


def test_frontier_invalid_covariance_returns_400(client):
    resp = client.get("/api/frontier", params={"covariance": "NOT_AN_ESTIMATOR"})
    assert resp.status_code == 400


def test_frontier_malformed_date_returns_422(client):
    resp = client.get("/api/frontier", params={"formation_date": "not-a-date"})
    assert resp.status_code == 422


def test_frontier_unknown_date_returns_404(client, monkeypatch):
    monkeypatch.setattr(frontier_module, "_official_formation_dates", lambda conn: [date(2022, 3, 28)])
    app.dependency_overrides[get_db] = _get_db_returning([])
    resp = client.get("/api/frontier", params={"formation_date": "2099-01-01", "covariance": "LW"})
    assert resp.status_code == 404


def test_frontier_no_official_dates_returns_404(client, monkeypatch):
    monkeypatch.setattr(frontier_module, "_official_formation_dates", lambda conn: [])
    app.dependency_overrides[get_db] = _get_db_returning([])
    resp = client.get("/api/frontier")
    assert resp.status_code == 404


def test_frontier_default_date_success_shape(client, monkeypatch):
    mu_arr, sigma_arr, rf_21 = _synthetic_frontier_context()
    monkeypatch.setattr(frontier_module, "_official_formation_dates", lambda conn: [date(2026, 1, 2)])
    monkeypatch.setattr(
        frontier_module, "reconstruct_mu_sigma_rf",
        lambda conn, fd: (mu_arr, {"SAMPLE": sigma_arr, "LW": sigma_arr}, rf_21),
    )
    monkeypatch.setattr(frontier_module, "_official_marker", _fake_official_marker)
    monkeypatch.setattr(frontier_module, "N_FRONTIER_POINTS", 5)
    app.dependency_overrides[get_db] = _get_db_returning([])

    resp = client.get("/api/frontier")
    assert resp.status_code == 200
    body = resp.json()
    assert body["formation_date"] == "2026-01-02"
    assert body["covariance_estimator"] == "Ledoit-Wolf"  # LW is the default
    assert body["forecast_horizon_sessions"] == 21
    assert body["covariance_window_sessions"] == 252
    assert body["max_weight_constraint"] == 0.10
    assert len(body["points"]) > 0
    for p in body["points"]:
        assert np.isfinite(p["expected_return_21"])
        assert p["volatility_21"] >= 0
    assert body["markers"]["min_vol"]["provenance"] == "official_phase7_persisted"
    assert body["markers"]["max_sharpe"]["provenance"] == "official_phase7_persisted"
    assert body["markers"]["equal_weight"]["provenance"] == "reconstructed_benchmark"
    assert body["source"] == "reconstructed_from_frozen_phase7_methodology"


def test_frontier_covariance_param_selects_matching_official_strategies(client, monkeypatch):
    mu_arr, sigma_arr, rf_21 = _synthetic_frontier_context()
    calls = []

    def _spy_marker(conn, formation_date, strategy, label):
        calls.append(strategy)
        return _fake_official_marker(conn, formation_date, strategy, label)

    monkeypatch.setattr(frontier_module, "_official_formation_dates", lambda conn: [date(2026, 1, 2)])
    monkeypatch.setattr(
        frontier_module, "reconstruct_mu_sigma_rf",
        lambda conn, fd: (mu_arr, {"SAMPLE": sigma_arr, "LW": sigma_arr}, rf_21),
    )
    monkeypatch.setattr(frontier_module, "_official_marker", _spy_marker)
    monkeypatch.setattr(frontier_module, "N_FRONTIER_POINTS", 3)
    app.dependency_overrides[get_db] = _get_db_returning([])

    resp = client.get("/api/frontier", params={"covariance": "SAMPLE"})
    assert resp.status_code == 200
    assert resp.json()["covariance_estimator"] == "Sample"
    assert calls == ["SAMPLE_MINVOL", "SAMPLE_MAXSHARPE"]


def test_frontier_equal_weight_marker_is_reconstructed_not_official(client, monkeypatch):
    n = len(config.TICKER_UNIVERSE)
    mu_arr = np.linspace(-0.01, 0.05, n)
    sigma_arr = np.eye(n) * 0.0004
    rf_21 = 0.001

    monkeypatch.setattr(frontier_module, "_official_formation_dates", lambda conn: [date(2026, 1, 2)])
    monkeypatch.setattr(
        frontier_module, "reconstruct_mu_sigma_rf",
        lambda conn, fd: (mu_arr, {"SAMPLE": sigma_arr, "LW": sigma_arr}, rf_21),
    )
    monkeypatch.setattr(frontier_module, "_official_marker", _fake_official_marker)
    monkeypatch.setattr(frontier_module, "N_FRONTIER_POINTS", 3)
    app.dependency_overrides[get_db] = _get_db_returning([])

    resp = client.get("/api/frontier")
    eq = resp.json()["markers"]["equal_weight"]
    assert eq["provenance"] == "reconstructed_benchmark"
    assert abs(eq["expected_return_21"] - float(np.mean(mu_arr))) < 1e-9
    expected_vol = float(np.sqrt(0.0004 / n))  # w=1/n, diagonal Sigma -> var = mean(diag)/n
    assert abs(eq["volatility_21"] - expected_vol) < 1e-9
    # Never implied to lie on the frontier — no such claim in the payload.
    assert "on_frontier" not in eq and "is_efficient" not in eq


def test_frontier_points_exclude_the_dominated_lower_branch(client, monkeypatch):
    """A page titled "Efficient Frontier" must never present the dominated
    (inefficient) branch of the minimum-variance parabola — for every
    returned point, no OTHER returned point has strictly lower volatility
    AND strictly higher return (the textbook non-domination property)."""
    mu_arr, sigma_arr, rf_21 = _synthetic_frontier_context()
    monkeypatch.setattr(frontier_module, "_official_formation_dates", lambda conn: [date(2026, 1, 2)])
    monkeypatch.setattr(
        frontier_module, "reconstruct_mu_sigma_rf",
        lambda conn, fd: (mu_arr, {"SAMPLE": sigma_arr, "LW": sigma_arr}, rf_21),
    )
    monkeypatch.setattr(frontier_module, "_official_marker", _fake_official_marker)
    monkeypatch.setattr(frontier_module, "N_FRONTIER_POINTS", 9)
    app.dependency_overrides[get_db] = _get_db_returning([])

    resp = client.get("/api/frontier")
    points = resp.json()["points"]
    assert len(points) > 1
    # Ascending target-return sweep -> volatility must be non-decreasing
    # too (monotonic upper branch), never dipping back down.
    vols = [p["volatility_21"] for p in points]
    assert vols == sorted(vols)
    rets = [p["expected_return_21"] for p in points]
    assert rets == sorted(rets)


def test_frontier_response_schema_is_lean_no_weight_matrices():
    from app.schemas import FrontierMarkers, FrontierPoint

    assert set(FrontierPoint.model_fields.keys()) == {"expected_return_21", "volatility_21", "sharpe_21"}
    assert set(FrontierMarkers.model_fields.keys()) == {"min_vol", "max_sharpe", "equal_weight"}  # no "spxt" field


def test_frontier_route_never_imports_stage_b_or_spxt_modules():
    """§20 causality guarantee, statically enforced: the frontier route's
    own import statements never reference Stage-B (realized-return) or
    SPXT (evaluation-only benchmark) code. Uses `ast` rather than a plain
    text search so the module's own explanatory docstring/comments (which
    legitimately mention these names to document the exclusion) can never
    cause a false failure."""
    path = pathlib.Path(__file__).resolve().parent.parent / "app" / "routes" / "frontier.py"
    tree = ast.parse(path.read_text())
    imported_names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            imported_names.add(module)
            imported_names.update(f"{module}.{alias.name}" for alias in node.names)
            imported_names.update(alias.name for alias in node.names)
    forbidden = {
        "optimizer.benchmark_spxt", "benchmark_spxt", "spxt_total_return",
        "optimizer.walkforward", "realized_stock_returns", "realized_portfolio_return",
    }
    hit = imported_names & forbidden
    assert not hit, f"frontier.py imports forbidden Stage-B/SPXT symbol(s): {hit}"


def test_frontier_module_namespace_has_no_stage_b_or_spxt_symbols():
    forbidden = {"realized_stock_returns", "realized_portfolio_return", "spxt_total_return", "spxt_raw"}
    names = set(vars(frontier_module).keys())
    assert not (names & forbidden), f"frontier module namespace unexpectedly binds: {names & forbidden}"


# ---------------------------------------------------------------------------
# Risk Analytics (formation-time component risk contribution — Phase 8D-1).
# Reuses `app.routes.frontier.reconstruct_mu_sigma_rf` for covariance, so the
# numeric reconciliation/causality proof lives in
# tests/test_api_db_integration.py against the real database. These are
# pure route-level unit tests: validation, wiring, response shape, and
# strategy/covariance provenance. Boundary functions
# (`_official_formation_dates`, `reconstruct_mu_sigma_rf`,
# `_load_official_weights`) are monkeypatched with small synthetic (but
# real-dimension) data.
# ---------------------------------------------------------------------------
def _equal_weights_50():
    n = len(config.TICKER_UNIVERSE)
    return np.full(n, 1.0 / n)


def _concentrated_optimized_weights_50():
    """10 names at exactly the 10% cap, the rest 0 — satisfies MAX_WEIGHT
    and sums to 1, matching the shape of a real optimized strategy."""
    n = len(config.TICKER_UNIVERSE)
    w = np.zeros(n)
    w[:10] = 0.10
    return w


def _synthetic_hedge_sigma():
    """A genuinely PSD 50x50 covariance where asset 0 has a NEGATIVE factor
    loading against the common factor everyone else shares — a "hedge"
    asset whose component risk contribution to an equal-weight portfolio
    is provably negative. Verified numerically before use (see the Phase
    8D-1 report): RC[0] < 0, sum(RC) == sigma_p, matrix PSD."""
    n = len(config.TICKER_UNIVERSE)
    b = np.full(n, 0.02)
    b[0] = -0.1
    return np.outer(b, b) + np.eye(n) * 1e-6


def test_risk_invalid_strategy_returns_400(client):
    resp = client.get("/api/risk", params={"strategy": "NOT_A_STRATEGY"})
    assert resp.status_code == 400


def test_risk_invalid_covariance_returns_400(client):
    resp = client.get("/api/risk", params={"covariance": "NOT_AN_ESTIMATOR"})
    assert resp.status_code == 400


def test_risk_malformed_date_returns_422(client):
    resp = client.get("/api/risk", params={"formation_date": "not-a-date"})
    assert resp.status_code == 422


def test_risk_unknown_date_returns_404(client, monkeypatch):
    monkeypatch.setattr(risk_module, "_official_formation_dates", lambda conn: [date(2022, 3, 28)])
    app.dependency_overrides[get_db] = _get_db_returning([])
    resp = client.get("/api/risk", params={"formation_date": "2099-01-01"})
    assert resp.status_code == 404


def test_risk_mismatched_covariance_for_optimized_strategy_returns_400(client, monkeypatch):
    """§7 — never let a client silently analyze an official optimized
    portfolio under a different covariance estimator than it was actually
    constructed under."""
    monkeypatch.setattr(risk_module, "_official_formation_dates", lambda conn: [date(2026, 1, 2)])
    app.dependency_overrides[get_db] = _get_db_returning([])
    resp = client.get("/api/risk", params={"strategy": "LW_MAXSHARPE", "covariance": "SAMPLE"})
    assert resp.status_code == 400
    assert "misrepresent" in resp.json()["detail"]


def test_risk_default_date_and_strategy_success_shape(client, monkeypatch):
    mu_arr, sigma_arr, rf_21 = _synthetic_frontier_context()
    w = _concentrated_optimized_weights_50()

    monkeypatch.setattr(risk_module, "_official_formation_dates", lambda conn: [date(2026, 1, 2)])
    monkeypatch.setattr(
        risk_module, "reconstruct_mu_sigma_rf",
        lambda conn, fd: (mu_arr, {"SAMPLE": sigma_arr, "LW": sigma_arr}, rf_21),
    )
    monkeypatch.setattr(risk_module, "_load_official_weights", lambda conn, fd, strat: w)
    app.dependency_overrides[get_db] = _get_db_returning([])

    resp = client.get("/api/risk")
    assert resp.status_code == 200
    body = resp.json()
    assert body["formation_date"] == "2026-01-02"
    assert body["strategy"]["key"] == "LW_MAXSHARPE"  # documented default
    assert body["covariance_estimator"] == "Ledoit-Wolf"  # implied by the default strategy
    assert body["forecast_horizon_sessions"] == 21
    assert body["covariance_window_sessions"] == 252
    assert body["portfolio"]["max_weight_constraint"] == 0.10
    assert len(body["assets"]) == 50
    assert {a["sector"] for a in body["assets"]} == {"Information Technology", "Financials"}
    assert len(body["sectors"]) == 2
    assert body["source"] == "reconstructed_from_frozen_phase7_methodology"

    # RC sum identity: sum_i RC_i == sigma_p (predicted_volatility_21).
    sigma_p = body["portfolio"]["predicted_volatility_21"]
    rc_sum = sum(a["component_risk_contribution"] for a in body["assets"])
    assert abs(rc_sum - sigma_p) < 1e-6

    # risk-share identity: sum_i risk_share_i == 1.
    share_sum = sum(a["risk_share"] for a in body["assets"])
    assert abs(share_sum - 1.0) < 1e-6

    # sector aggregation identity: sum of sector RC == total RC == sigma_p.
    sector_rc_sum = sum(s["component_risk_contribution"] for s in body["sectors"])
    assert abs(sector_rc_sum - sigma_p) < 1e-6
    sector_weight_sum = sum(s["weight"] for s in body["sectors"])
    assert abs(sector_weight_sum - 1.0) < 1e-6


def test_risk_no_raw_covariance_matrix_in_payload(client, monkeypatch):
    mu_arr, sigma_arr, rf_21 = _synthetic_frontier_context()
    w = _concentrated_optimized_weights_50()
    monkeypatch.setattr(risk_module, "_official_formation_dates", lambda conn: [date(2026, 1, 2)])
    monkeypatch.setattr(
        risk_module, "reconstruct_mu_sigma_rf",
        lambda conn, fd: (mu_arr, {"SAMPLE": sigma_arr, "LW": sigma_arr}, rf_21),
    )
    monkeypatch.setattr(risk_module, "_load_official_weights", lambda conn, fd, strat: w)
    app.dependency_overrides[get_db] = _get_db_returning([])

    resp = client.get("/api/risk")
    body = resp.json()
    dumped = str(body)
    assert "sigma" not in dumped.lower().replace("sigma_p", "")  # no raw covariance leaked into the response


def test_risk_equal_weight_defaults_to_ledoit_wolf_and_uses_reconstructed_context(client, monkeypatch):
    n = len(config.TICKER_UNIVERSE)
    mu_arr = np.linspace(-0.01, 0.05, n)
    sigma_arr = np.eye(n) * 0.0004
    rf_21 = 0.001
    w = _equal_weights_50()

    monkeypatch.setattr(risk_module, "_official_formation_dates", lambda conn: [date(2026, 1, 2)])
    monkeypatch.setattr(
        risk_module, "reconstruct_mu_sigma_rf",
        lambda conn, fd: (mu_arr, {"SAMPLE": sigma_arr, "LW": sigma_arr}, rf_21),
    )
    monkeypatch.setattr(risk_module, "_load_official_weights", lambda conn, fd, strat: w)
    app.dependency_overrides[get_db] = _get_db_returning([])

    resp = client.get("/api/risk", params={"strategy": "EQUAL_WEIGHT"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["covariance_estimator"] == "Ledoit-Wolf"  # default when unspecified
    assert body["portfolio"]["max_weight_constraint"] is None  # never routed through MAX_WEIGHT
    assert all(abs(a["weight"] - 1.0 / n) < 1e-9 for a in body["assets"])  # exact conceptual 1/50, never reoptimized


def test_risk_equal_weight_explicit_sample_covariance_is_analysis_only(client, monkeypatch):
    n = len(config.TICKER_UNIVERSE)
    mu_arr = np.linspace(-0.01, 0.05, n)
    sample_sigma = np.eye(n) * 0.0004
    lw_sigma = np.eye(n) * 0.0009  # deliberately different, to prove the selector actually switches
    rf_21 = 0.001
    w = _equal_weights_50()

    monkeypatch.setattr(risk_module, "_official_formation_dates", lambda conn: [date(2026, 1, 2)])
    monkeypatch.setattr(
        risk_module, "reconstruct_mu_sigma_rf",
        lambda conn, fd: (mu_arr, {"SAMPLE": sample_sigma, "LW": lw_sigma}, rf_21),
    )
    monkeypatch.setattr(risk_module, "_load_official_weights", lambda conn, fd, strat: w)
    app.dependency_overrides[get_db] = _get_db_returning([])

    resp = client.get("/api/risk", params={"strategy": "EQUAL_WEIGHT", "covariance": "SAMPLE"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["covariance_estimator"] == "Sample"
    # Equal-weight portfolio variance under a diagonal Sigma=var*I is var/n;
    # confirms the SAMPLE selection actually took effect (would be
    # sqrt(0.0009/n) if LW had been used instead).
    import math
    assert abs(body["portfolio"]["predicted_volatility_21"] - math.sqrt(0.0004 / n)) < 1e-9


def test_risk_negative_component_contribution_is_preserved_not_clamped(client, monkeypatch):
    """§4 — a genuine hedge asset can carry a NEGATIVE component risk
    contribution; it must never be clamped to zero, abs'd, or
    renormalized away."""
    n = len(config.TICKER_UNIVERSE)
    mu_arr = np.zeros(n)
    sigma_arr = _synthetic_hedge_sigma()
    rf_21 = 0.001
    w = _equal_weights_50()

    monkeypatch.setattr(risk_module, "_official_formation_dates", lambda conn: [date(2026, 1, 2)])
    monkeypatch.setattr(
        risk_module, "reconstruct_mu_sigma_rf",
        lambda conn, fd: (mu_arr, {"SAMPLE": sigma_arr, "LW": sigma_arr}, rf_21),
    )
    monkeypatch.setattr(risk_module, "_load_official_weights", lambda conn, fd, strat: w)
    app.dependency_overrides[get_db] = _get_db_returning([])

    resp = client.get("/api/risk", params={"strategy": "EQUAL_WEIGHT"})
    assert resp.status_code == 200
    body = resp.json()
    hedge_asset = next(a for a in body["assets"] if a["ticker"] == config.TICKER_UNIVERSE[0])
    assert hedge_asset["component_risk_contribution"] < 0
    assert hedge_asset["risk_share"] < 0
    # Identity must still hold even with a negative term present.
    sigma_p = body["portfolio"]["predicted_volatility_21"]
    rc_sum = sum(a["component_risk_contribution"] for a in body["assets"])
    assert abs(rc_sum - sigma_p) < 1e-6


def test_risk_response_schema_has_no_ex_post_or_spxt_fields():
    from app.schemas import AssetRiskRow, PortfolioRiskSummary, RiskResponse

    forbidden = {"realized_return_21", "turnover", "spxt", "actual_return"}
    for schema in (AssetRiskRow, PortfolioRiskSummary, RiskResponse):
        assert not (set(schema.model_fields.keys()) & forbidden), f"{schema.__name__} leaks an ex-post/SPXT field"


def test_risk_route_never_imports_stage_b_or_spxt_modules():
    """Same AST-based static causality guarantee as the Frontier route:
    Risk Analytics is entirely ex-ante, so Stage-B (realized-return) and
    SPXT (evaluation-only benchmark) symbols must never be imported."""
    path = pathlib.Path(__file__).resolve().parent.parent / "app" / "routes" / "risk.py"
    tree = ast.parse(path.read_text())
    imported_names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            imported_names.add(module)
            imported_names.update(f"{module}.{alias.name}" for alias in node.names)
            imported_names.update(alias.name for alias in node.names)
    forbidden = {
        "optimizer.benchmark_spxt", "benchmark_spxt", "spxt_total_return",
        "optimizer.walkforward", "realized_stock_returns", "realized_portfolio_return",
    }
    hit = imported_names & forbidden
    assert not hit, f"risk.py imports forbidden Stage-B/SPXT symbol(s): {hit}"


def test_risk_module_namespace_has_no_stage_b_or_spxt_symbols():
    forbidden = {"realized_stock_returns", "realized_portfolio_return", "spxt_total_return", "spxt_raw"}
    names = set(vars(risk_module).keys())
    assert not (names & forbidden), f"risk module namespace unexpectedly binds: {names & forbidden}"


# ---------------------------------------------------------------------------
# Historical Evidence (frozen, official Phase 7 walk-forward experiment —
# Phase 8D-2). UNLIKE Frontier/Risk, this route legitimately imports
# `optimizer.walkforward` (aggregation/compounding helpers) and
# `optimizer.benchmark_spxt` (the official benchmark) — it is specifically
# about presenting realized/evaluation results. What it must NEVER do is
# reconstruct realized figures with a second formula, use the deprecated
# non-official SPX diagnostic, or touch sealed/extension data. Numeric
# reconciliation against the frozen Phase 7 report lives in
# tests/test_api_db_integration.py; these are route-level unit tests using
# a small synthetic (but real-shaped, 46-period, sequentially-chained)
# calendar so the real `aggregate_statistics`/`cumulative_compounded_return`
# math still runs, without touching the live database or
# data/raw/spxt_benchmark.csv.
# ---------------------------------------------------------------------------
from optimizer.walkforward import StrategyPeriodResult as _StrategyPeriodResult  # noqa: E402


def _synthetic_backtest_calendar(n=46):
    dates = pd.date_range("2022-01-03", periods=n + 1, freq="21D")
    return pd.DataFrame({"forecast_date": dates[:-1], "target_date": dates[1:]})


def _synthetic_period_results(calendar_df, returns, turnovers):
    return [
        _StrategyPeriodResult(formation_date=fd, realized_return=r, turnover=t, max_weight_observed=0.1)
        for fd, r, t in zip(calendar_df["forecast_date"], returns, turnovers)
    ]


def _get_db_returning_data_through(d=date(2026, 2, 27)):
    """The backtest route's final query (`SELECT MAX(date) FROM
    prices_raw`) needs `fetchone()` to return a real one-tuple row, unlike
    the plain `_get_db_returning([])` used by routes that never reach a
    trailing scalar query."""
    return _get_db_returning([(d,)])


def _patch_backtest_boundaries(monkeypatch, calendar_df, returns_by_strategy, spxt_series):
    n = len(calendar_df)
    monkeypatch.setattr(backtest_module, "_official_formation_dates", lambda conn: [date(2022, 1, 1)] * n)
    monkeypatch.setattr(backtest_module, "_load_formation_target_calendar", lambda conn, dates: calendar_df)

    def _fake_period_results(conn, strategy, cal):
        returns, turnovers = returns_by_strategy[strategy]
        return _synthetic_period_results(cal, returns, turnovers)

    monkeypatch.setattr(backtest_module, "_load_strategy_period_results", _fake_period_results)
    monkeypatch.setattr(backtest_module, "_spxt_series", lambda cal: spxt_series)


def _default_returns_by_strategy(n=46):
    from optimizer.persistence import VALID_STRATEGIES

    out = {}
    for i, strat in enumerate(VALID_STRATEGIES):
        returns = [0.01 * ((j % 5) - 2) for j in range(n)]  # deterministic, in [-0.02, 0.02]
        turnovers = [float("nan")] + [0.1] * (n - 1)  # first undefined, matches real persisted behavior
        out[strat] = (returns, turnovers)
    return out


def _fake_spxt_series(calendar_df):
    from app.schemas import BacktestPeriod, BacktestSeries, BacktestSummary

    returns = [0.005] * len(calendar_df)
    growth = []
    level = 1.0
    for r in returns:
        level *= 1 + r
        growth.append(level)
    periods = [
        BacktestPeriod(
            formation_date=row["forecast_date"].date(),
            target_date=row["target_date"].date(),
            realized_return_21=returns[i],
            growth_of_one=growth[i],
            turnover=None,
            max_weight_observed=None,
        )
        for i, (_, row) in enumerate(calendar_df.iterrows())
    ]
    return BacktestSeries(
        key="SPXT", label="SPXT (S&P 500 Total Return)", kind="benchmark",
        is_optimized=None, covariance_estimator=None, max_weight_constraint=None,
        periods=periods,
        summary=BacktestSummary(
            mean_return_21=0.005, std_return_21=0.0, median_return_21=0.005, min_return_21=0.005,
            max_return_21=0.005, positive_period_rate=1.0, cumulative_return=growth[-1] - 1.0,
        ),
    )


def test_backtest_returns_six_series_with_46_periods_each(client, monkeypatch):
    calendar_df = _synthetic_backtest_calendar()
    _patch_backtest_boundaries(monkeypatch, calendar_df, _default_returns_by_strategy(), _fake_spxt_series(calendar_df))
    app.dependency_overrides[get_db] = _get_db_returning_data_through()

    resp = client.get("/api/backtest")
    assert resp.status_code == 200
    body = resp.json()
    assert body["experiment"]["period_count"] == 46
    assert len(body["series"]) == 6
    keys = {s["key"] for s in body["series"]}
    assert keys == {"SAMPLE_MINVOL", "SAMPLE_MAXSHARPE", "LW_MINVOL", "LW_MAXSHARPE", "EQUAL_WEIGHT", "SPXT"}
    for s in body["series"]:
        assert len(s["periods"]) == 46


def test_backtest_kind_and_provenance_flags_are_truthful(client, monkeypatch):
    calendar_df = _synthetic_backtest_calendar()
    _patch_backtest_boundaries(monkeypatch, calendar_df, _default_returns_by_strategy(), _fake_spxt_series(calendar_df))
    app.dependency_overrides[get_db] = _get_db_returning_data_through()

    resp = client.get("/api/backtest")
    body = resp.json()
    by_key = {s["key"]: s for s in body["series"]}
    for key in ("SAMPLE_MINVOL", "SAMPLE_MAXSHARPE", "LW_MINVOL", "LW_MAXSHARPE"):
        assert by_key[key]["kind"] == "portfolio"
        assert by_key[key]["is_optimized"] is True
        assert by_key[key]["max_weight_constraint"] == 0.10
    assert by_key["EQUAL_WEIGHT"]["kind"] == "portfolio"
    assert by_key["EQUAL_WEIGHT"]["is_optimized"] is False
    assert by_key["EQUAL_WEIGHT"]["max_weight_constraint"] is None
    assert by_key["SPXT"]["kind"] == "benchmark"
    assert by_key["SPXT"]["is_optimized"] is None
    assert by_key["SPXT"]["covariance_estimator"] is None
    assert by_key["SPXT"]["max_weight_constraint"] is None


def test_backtest_first_period_turnover_is_null_never_zero_for_portfolios(client, monkeypatch):
    calendar_df = _synthetic_backtest_calendar()
    _patch_backtest_boundaries(monkeypatch, calendar_df, _default_returns_by_strategy(), _fake_spxt_series(calendar_df))
    app.dependency_overrides[get_db] = _get_db_returning_data_through()

    resp = client.get("/api/backtest")
    body = resp.json()
    for s in body["series"]:
        if s["kind"] != "portfolio":
            continue
        assert s["periods"][0]["turnover"] is None
        assert s["periods"][1]["turnover"] is not None  # a real, later persisted value


def test_backtest_spxt_series_has_no_turnover_or_max_weight(client, monkeypatch):
    calendar_df = _synthetic_backtest_calendar()
    _patch_backtest_boundaries(monkeypatch, calendar_df, _default_returns_by_strategy(), _fake_spxt_series(calendar_df))
    app.dependency_overrides[get_db] = _get_db_returning_data_through()

    resp = client.get("/api/backtest")
    spxt = next(s for s in resp.json()["series"] if s["key"] == "SPXT")
    assert spxt["summary"]["mean_turnover"] is None
    assert spxt["summary"]["avg_max_weight"] is None
    assert all(p["turnover"] is None and p["max_weight_observed"] is None for p in spxt["periods"])


def test_backtest_rejects_a_period_count_other_than_46(client, monkeypatch):
    """§15 period-count verification: the route must refuse to serve a
    result claiming to be the official Phase 7 experiment if the eligible
    formation calendar isn't exactly 46 dates — never silently proceed
    with a different count."""
    calendar_df = _synthetic_backtest_calendar(n=3)
    from optimizer.persistence import VALID_STRATEGIES

    returns_by_strategy = {strat: ([0.01, -0.02, 0.03], [float("nan"), 0.1, 0.1]) for strat in VALID_STRATEGIES}
    _patch_backtest_boundaries(monkeypatch, calendar_df, returns_by_strategy, _fake_spxt_series(calendar_df))
    app.dependency_overrides[get_db] = _get_db_returning([])
    monkeypatch.setattr(backtest_module, "_official_formation_dates", lambda conn: [date(2022, 1, 1)] * 3)

    resp = client.get("/api/backtest")
    assert resp.status_code == 500
    assert "46" in resp.json()["detail"]


def test_backtest_growth_of_one_matches_hand_computed_compounding(client, monkeypatch):
    """Same compounding check as above, but at the full 46-period shape
    the route requires, so it actually returns 200 — verifies growth_of_one
    at every step equals sequential compounding, and the final growth
    matches the reported cumulative_return (never sum(returns))."""
    calendar_df = _synthetic_backtest_calendar(n=46)
    returns = [0.01, -0.02, 0.03] + [0.0] * 43
    turnovers = [float("nan")] + [0.1] * 45
    from optimizer.persistence import VALID_STRATEGIES

    returns_by_strategy = {strat: (returns, turnovers) for strat in VALID_STRATEGIES}
    _patch_backtest_boundaries(monkeypatch, calendar_df, returns_by_strategy, _fake_spxt_series(calendar_df))
    app.dependency_overrides[get_db] = _get_db_returning_data_through()

    resp = client.get("/api/backtest")
    assert resp.status_code == 200
    series = next(s for s in resp.json()["series"] if s["key"] == "LW_MAXSHARPE")
    periods = series["periods"]

    expected_growth = []
    level = 1.0
    for r in returns:
        level *= 1 + r
        expected_growth.append(level)

    for i in (0, 1, 2):
        assert abs(periods[i]["growth_of_one"] - expected_growth[i]) < 1e-9
    # Flat thereafter (0% returns) — growth must stay constant, not drift.
    assert abs(periods[-1]["growth_of_one"] - expected_growth[-1]) < 1e-9

    # Cumulative return must equal the compounded product minus 1 —
    # never the naive sum of the 46 period returns.
    assert abs(series["summary"]["cumulative_return"] - (expected_growth[-1] - 1.0)) < 1e-9
    assert abs(series["summary"]["cumulative_return"] - sum(returns)) > 1e-6


def test_backtest_response_schema_is_lean_no_raw_covariance_or_holdings():
    from app.schemas import BacktestPeriod, BacktestSeries

    period_fields = set(BacktestPeriod.model_fields.keys())
    assert "weights" not in period_fields and "holdings" not in period_fields
    series_fields = set(BacktestSeries.model_fields.keys())
    assert "weights" not in series_fields and "assets" not in series_fields


def test_backtest_route_never_reconstructs_realized_returns_or_uses_deprecated_spx_diagnostic():
    """Backend causality guard specific to this route: it must read
    already-persisted `realized_return_21` (never recompute it via
    Stage-B's raw stock/portfolio-return reconstruction) and must use only
    the official SPXT total-return helper — never the deprecated,
    non-official SPX price-return diagnostic."""
    path = pathlib.Path(__file__).resolve().parent.parent / "app" / "routes" / "backtest.py"
    tree = ast.parse(path.read_text())
    imported_names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            imported_names.add(module)
            imported_names.update(f"{module}.{alias.name}" for alias in node.names)
            imported_names.update(alias.name for alias in node.names)
    forbidden = {"realized_stock_returns", "realized_portfolio_return", "spx_price_return_diagnostic"}
    hit = imported_names & forbidden
    assert not hit, f"backtest.py imports forbidden symbol(s): {hit}"


def test_backtest_route_never_references_sealed_or_extension_data():
    """The module docstring legitimately DOCUMENTS this exclusion (naming
    the forbidden files to explain why they're absent) — strip it before
    scanning so that documentation can't itself trip the guard; only the
    executable code body is checked."""
    path = pathlib.Path(__file__).resolve().parent.parent / "app" / "routes" / "backtest.py"
    source = path.read_text()
    module_docstring = ast.get_docstring(ast.parse(source)) or ""
    code_only = source.replace(module_docstring, "").lower()
    for forbidden in ("prices_sealed", "macro_sealed", "prices_extension", "macro_extension"):
        assert forbidden not in code_only, f"backtest.py references forbidden sealed/extension artifact: {forbidden}"


# ---------------------------------------------------------------------------
# Research API is read-only: no mutating SQL keyword anywhere in app/routes/
# ---------------------------------------------------------------------------
def test_research_routes_never_issue_mutating_sql():
    routes_dir = pathlib.Path(__file__).resolve().parent.parent / "app" / "routes"
    for path in routes_dir.glob("*.py"):
        text = path.read_text().lower()
        assert not _SQL_MUTATION_KEYWORDS.search(text), f"{path} appears to contain mutating SQL"


# ---------------------------------------------------------------------------
# Not-yet-authorized research tables must still never be referenced in app/
# code. `predictions` (Phase 8B Forecast Rankings/Model Comparison) and
# `portfolios`/`risk_metrics` (Phase 8C Portfolio Construction —
# app/routes/portfolios.py reads both) are now authorized and deliberately
# removed from this list. `features` remains unauthorized — no current
# surface (Efficient Frontier, Risk Analytics, Historical Evidence) needs it
# yet.
# ---------------------------------------------------------------------------
def test_unauthorized_future_phase_tables_never_referenced_in_app_code():
    forbidden_tables = ["features"]
    app_dir = pathlib.Path(__file__).resolve().parent.parent / "app"
    for path in app_dir.rglob("*.py"):
        text = path.read_text().lower()
        for table in forbidden_tables:
            assert table not in text, f"{path} references not-yet-authorized table '{table}'"
