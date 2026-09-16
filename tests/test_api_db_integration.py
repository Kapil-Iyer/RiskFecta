"""
Phase 2A local-only integration tests against the REAL Supabase-hosted
PostgreSQL database, exercised through the actual FastAPI app (not mocks).
Follows the same pattern as tests/test_db_integration.py: every test here is
skipped automatically when DATABASE_URL is not set (e.g. in CI). Run locally
with: pytest -m db

These tests never print, log, or assert on the DATABASE_URL value itself.
"""
from __future__ import annotations

import os

import pytest
from fastapi.testclient import TestClient

import config
from app.main import app
from pipeline import db

pytestmark = pytest.mark.db

_skip_no_db = pytest.mark.skipif(
    not os.environ.get("DATABASE_URL"),
    reason="DATABASE_URL not set in the environment (expected in CI / clean checkouts)",
)

# Downstream research tables the read-only API must never mutate.
# `features` (Phase 3) and `predictions` (Phase 4) were already populated
# even at Phase 2A. `portfolios`/`risk_metrics` (Phase 7) were empty when
# this test was first written but now legitimately hold the official,
# frozen Phase 7B experiment (11,500 / 685 rows) — so, like
# features/predictions, the invariant is "row count unchanged across API
# calls", never a hard-coded emptiness assumption
# (see test_future_phase_tables_remain_unchanged_after_api_use). Phase 8B
# adds real GET endpoints over `predictions` (see test_prediction_* below) —
# that table's row count must still never change from ordinary API reads.
UNCHANGED_DOWNSTREAM_TABLES = ["features", "predictions", "portfolios", "risk_metrics"]
EXPECTED_ROW_COUNT = 62_800  # 50 tickers x 1256 valid sessions each (Phase 1A audit)

# Frozen Phase 4-6 forecasting calendar (BUILD_PLAN.md Phase 4-6, ML_SPEC.md §23).
EXPECTED_PREDICTION_DATE_COUNT = 47
EXPECTED_FIRST_FORMATION_DATE = "2022-02-25"
EXPECTED_LAST_FORMATION_DATE = "2026-01-02"


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


@_skip_no_db
def test_health_ready_reports_connected(client):
    resp = client.get("/health/ready")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok", "database": "connected"}


@_skip_no_db
def test_universe_matches_config_universe(client):
    resp = client.get("/api/universe")
    assert resp.status_code == 200
    body = resp.json()
    assert {row["ticker"] for row in body} == set(config.TICKER_UNIVERSE)


@_skip_no_db
def test_prices_real_ticker_returns_full_ordered_history(client):
    resp = client.get("/api/prices/AAPL")
    assert resp.status_code == 200
    body = resp.json()
    assert body["ticker"] == "AAPL"
    assert body["count"] == 1256
    dates = [row["date"] for row in body["prices"]]
    assert dates == sorted(dates)


@_skip_no_db
def test_prices_date_filter_narrows_range(client):
    resp = client.get("/api/prices/AAPL", params={"start": "2025-01-01", "end": "2025-01-31"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["count"] > 0
    assert all("2025-01-01" <= row["date"] <= "2025-01-31" for row in body["prices"])


@_skip_no_db
def test_prices_unknown_ticker_returns_404(client):
    resp = client.get("/api/prices/ZZZNOTAREALTICKER")
    assert resp.status_code == 404


@_skip_no_db
def test_prices_invalid_date_range_returns_400(client):
    resp = client.get("/api/prices/AAPL", params={"start": "2025-02-01", "end": "2025-01-01"})
    assert resp.status_code == 400


@_skip_no_db
def test_market_summary_matches_known_totals(client):
    resp = client.get("/api/market/summary")
    assert resp.status_code == 200
    body = resp.json()
    assert body["ticker_count"] == 50
    assert body["price_row_count"] == EXPECTED_ROW_COUNT
    assert body["first_date"] == "2021-03-01"
    assert body["last_date"] == "2026-02-27"


@_skip_no_db
def test_prediction_dates_returns_all_47_frozen_formations(client):
    resp = client.get("/api/predictions/dates")
    assert resp.status_code == 200
    dates = resp.json()
    assert len(dates) == EXPECTED_PREDICTION_DATE_COUNT
    assert dates[0] == EXPECTED_FIRST_FORMATION_DATE
    assert dates[-1] == EXPECTED_LAST_FORMATION_DATE
    assert dates == sorted(dates)


@_skip_no_db
def test_predictions_default_returns_latest_formation_with_50_tickers(client):
    resp = client.get("/api/predictions")
    assert resp.status_code == 200
    body = resp.json()
    assert body["formation_date"] == EXPECTED_LAST_FORMATION_DATE
    assert body["count"] == 50
    assert len(body["predictions"]) == 50
    tickers = [row["ticker"] for row in body["predictions"]]
    assert tickers == sorted(tickers)  # base order is deterministic: ticker ascending
    assert set(tickers) == set(config.TICKER_UNIVERSE)


@_skip_no_db
def test_predictions_specific_formation_returns_50_rows_with_all_models(client):
    # First covariance-eligible Phase 7 formation — a real, known-valid date.
    resp = client.get("/api/predictions", params={"formation_date": "2022-03-28"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["formation_date"] == "2022-03-28"
    assert body["count"] == 50
    for row in body["predictions"]:
        assert row["xgb_pred"] is not None
        assert row["lstm_pred"] is not None
        assert row["ensemble_pred"] is not None
        # 21 sessions after 2022-03-28 is well before the 2026-02-27 data
        # cutoff, so the realized outcome should already be evaluated.
        assert row["actual_return"] is not None
        assert row["directional_correct"] is not None


@_skip_no_db
def test_predictions_unknown_formation_date_returns_404(client):
    resp = client.get("/api/predictions", params={"formation_date": "2099-01-01"})
    assert resp.status_code == 404


@_skip_no_db
def test_model_comparison_returns_six_models_with_correct_provenance(client):
    resp = client.get("/api/models/comparison")
    assert resp.status_code == 200
    body = resp.json()
    assert body["fold_count"] == EXPECTED_PREDICTION_DATE_COUNT
    assert body["prediction_count"] == 2350
    assert body["formation_date_start"] == EXPECTED_FIRST_FORMATION_DATE
    assert body["formation_date_end"] == EXPECTED_LAST_FORMATION_DATE

    models_by_key = {m["model"]: m for m in body["models"]}
    assert set(models_by_key) == {"historical_mean", "momentum_3m", "ridge", "xgb_pred", "lstm_pred", "ensemble_pred"}
    for key in ("historical_mean", "momentum_3m", "ridge"):
        assert models_by_key[key]["source"] == "frozen_baseline_constant"
        assert models_by_key[key]["n_obs"] == 2350
    for key in ("xgb_pred", "lstm_pred", "ensemble_pred"):
        assert models_by_key[key]["source"] == "computed_from_persisted_predictions"
        assert models_by_key[key]["n_obs"] == 2350


@_skip_no_db
def test_model_comparison_baseline_values_match_frozen_constants(client):
    from models.frozen_phase4_baselines import FROZEN_BASELINE_RESULTS

    resp = client.get("/api/models/comparison")
    models_by_key = {m["model"]: m for m in resp.json()["models"]}
    for expected in FROZEN_BASELINE_RESULTS:
        actual = models_by_key[expected.model]
        assert actual["mae"] == expected.mae
        assert actual["rmse"] == expected.rmse
        assert actual["directional_accuracy"] == expected.directional_accuracy
        assert actual["pearson_corr"] == expected.pearson_corr
        assert actual["spearman_corr"] == expected.spearman_corr


@_skip_no_db
def test_model_comparison_ml_metrics_reproduce_authoritative_phase6_results(client):
    """Independently confirms the live route (recomputing from persisted
    `predictions` via the frozen evaluation code) reproduces the
    authoritative, already-established Phase 6 walk-forward results —
    validation of an existing frozen result, never a new experiment. A
    mismatch here is an integrity issue to investigate, not something to
    silently reconcile by editing either side."""
    # Authoritative Phase 6 historical walk-forward OOS results (47 dates,
    # 2,350 predictions) — independently confirmed to 5 decimal places
    # against a fresh recomputation from `predictions` before this route
    # was written (see the Phase 8B implementation report).
    expected = {
        "xgb_pred": dict(mae=0.07915, rmse=0.10453, directional_accuracy=0.4996, pearson_corr=0.07105, spearman_corr=0.00007),
        "lstm_pred": dict(mae=0.08156, rmse=0.11030, directional_accuracy=0.5085, pearson_corr=0.02197, spearman_corr=0.00805),
        "ensemble_pred": dict(mae=0.07717, rmse=0.10292, directional_accuracy=0.50724, pearson_corr=0.05479, spearman_corr=0.01721),
    }
    resp = client.get("/api/models/comparison")
    models_by_key = {m["model"]: m for m in resp.json()["models"]}
    for model, metrics in expected.items():
        actual = models_by_key[model]
        for metric, value in metrics.items():
            assert abs(actual[metric] - value) < 1e-4, f"{model}.{metric}: {actual[metric]} vs frozen {value}"


@_skip_no_db
def test_model_comparison_disagreement_matches_phase6_diagnostics(client):
    resp = client.get("/api/models/comparison")
    disagreement = resp.json()["disagreement"]
    assert disagreement is not None
    assert disagreement["n_total"] == 2350
    assert disagreement["n_disagree"] == 773
    assert abs(disagreement["xgb_lstm_pred_pearson"] - 0.341) < 1e-2
    assert abs(disagreement["residual_pearson"] - 0.834) < 1e-2


@_skip_no_db
def test_future_phase_tables_remain_unchanged_after_api_use(client):
    """Phase 2A read-only API must not mutate later-phase research tables.

    Every downstream table's row count must be identical before and after
    a batch of ordinary read-only API calls — never a hard-coded
    emptiness assumption, which breaks the moment a later phase
    legitimately populates that table (as Phase 7B's frozen 46-period
    experiment now has for `portfolios`/`risk_metrics`).
    """
    conn = db.get_connection()
    try:
        before = {t: db.fetch_scalar(conn, f"SELECT COUNT(*) FROM {t}") for t in UNCHANGED_DOWNSTREAM_TABLES}
    finally:
        conn.close()

    client.get("/health")
    client.get("/health/ready")
    client.get("/api/universe")
    client.get("/api/prices/AAPL")
    client.get("/api/prices/AAPL", params={"start": "2025-01-01", "end": "2025-01-31"})
    client.get("/api/market/summary")
    client.get("/api/predictions/dates")
    client.get("/api/predictions")
    client.get("/api/predictions", params={"formation_date": "2022-03-28"})
    client.get("/api/models/comparison")

    conn = db.get_connection()
    try:
        for table, before_count in before.items():
            after_count = db.fetch_scalar(conn, f"SELECT COUNT(*) FROM {table}")
            assert after_count == before_count, (
                f"{table} row count changed across Phase 2A API use ({before_count} -> {after_count})"
            )
    finally:
        conn.close()
