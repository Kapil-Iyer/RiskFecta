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

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

import config
from app.main import app
from app.routes import frontier, risk
from optimizer.persistence import build_run_id
from optimizer.portfolio import portfolio_expected_return, portfolio_sharpe, portfolio_volatility
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

# Frozen Phase 7 official portfolio experiment (BUILD_PLAN.md Phase 7).
EXPECTED_PORTFOLIO_DATE_COUNT = 46
EXPECTED_FIRST_PORTFOLIO_DATE = "2022-03-28"
EXPECTED_LAST_PORTFOLIO_DATE = "2026-01-02"
EXPECTED_STRATEGIES = ("SAMPLE_MINVOL", "SAMPLE_MAXSHARPE", "LW_MINVOL", "LW_MAXSHARPE", "EQUAL_WEIGHT")


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
def test_portfolio_dates_are_exactly_the_46_official_formations(client):
    resp = client.get("/api/portfolios/dates")
    assert resp.status_code == 200
    dates = resp.json()
    assert len(dates) == EXPECTED_PORTFOLIO_DATE_COUNT
    assert dates[0] == EXPECTED_FIRST_PORTFOLIO_DATE
    assert dates[-1] == EXPECTED_LAST_PORTFOLIO_DATE
    assert dates == sorted(dates)
    assert "2022-02-25" not in dates  # prediction-only, never portfolio-eligible


@_skip_no_db
def test_portfolio_strategies_are_exactly_the_five_official_strategies(client):
    resp = client.get("/api/portfolios/strategies")
    assert resp.status_code == 200
    assert tuple(s["key"] for s in resp.json()) == EXPECTED_STRATEGIES


@_skip_no_db
def test_portfolio_default_is_latest_date_and_lw_maxsharpe_with_50_holdings(client):
    resp = client.get("/api/portfolios")
    assert resp.status_code == 200
    body = resp.json()
    assert body["formation_date"] == EXPECTED_LAST_PORTFOLIO_DATE
    assert body["strategy"]["key"] == "LW_MAXSHARPE"
    assert len(body["holdings"]) == 50
    tickers = [h["ticker"] for h in body["holdings"]]
    assert tickers == sorted(tickers)  # deterministic ticker-ascending base order
    assert set(tickers) == set(config.TICKER_UNIVERSE)


@_skip_no_db
@pytest.mark.parametrize(
    "formation_date,strategy",
    [
        (EXPECTED_LAST_PORTFOLIO_DATE, "LW_MAXSHARPE"),
        (EXPECTED_LAST_PORTFOLIO_DATE, "LW_MINVOL"),
        (EXPECTED_LAST_PORTFOLIO_DATE, "EQUAL_WEIGHT"),
        (EXPECTED_FIRST_PORTFOLIO_DATE, "SAMPLE_MAXSHARPE"),
        (EXPECTED_LAST_PORTFOLIO_DATE, "SAMPLE_MINVOL"),
    ],
)
def test_portfolio_real_combinations_satisfy_frozen_constraints(client, formation_date, strategy):
    resp = client.get("/api/portfolios", params={"formation_date": formation_date, "strategy": strategy})
    assert resp.status_code == 200
    body = resp.json()

    weights = [h["weight"] for h in body["holdings"]]
    assert len(weights) == 50
    assert all(w >= -1e-9 for w in weights)  # no negative weights
    assert abs(sum(weights) - 1.0) < 1e-4  # fully invested

    if strategy == "EQUAL_WEIGHT":
        assert all(abs(w - 1.0 / 50) < 1e-6 for w in weights)  # exact conceptual 2% each
        assert body["max_weight_constraint"] is None
        assert body["construction"] == {"expected_return_21": None, "predicted_volatility_21": None, "expected_sharpe_21": None}
    else:
        assert all(w <= 0.10 + 1e-6 for w in weights)  # hard 10% cap, storage-precision tolerance
        assert body["max_weight_constraint"] == 0.10
        assert body["construction"]["expected_return_21"] is not None
        assert body["construction"]["predicted_volatility_21"] is not None
        assert body["construction"]["expected_sharpe_21"] is not None

    # Ex-post evaluation is always present and clearly separate from construction.
    assert isinstance(body["evaluation"]["realized_return_21"], float)
    assert isinstance(body["evaluation"]["max_weight_observed"], float)

    # Derived summary values are internally consistent with the weights.
    assert abs(body["largest_weight"] - max(weights)) < 1e-9
    assert abs(body["concentration_hhi"] - sum(w * w for w in weights)) < 1e-9


@_skip_no_db
def test_portfolio_first_official_formation_has_undefined_turnover(client):
    resp = client.get("/api/portfolios", params={"formation_date": EXPECTED_FIRST_PORTFOLIO_DATE, "strategy": "LW_MAXSHARPE"})
    assert resp.status_code == 200
    assert resp.json()["evaluation"]["turnover"] is None  # never 0 — genuinely undefined


@_skip_no_db
def test_portfolio_later_formation_has_defined_turnover(client):
    resp = client.get("/api/portfolios", params={"formation_date": EXPECTED_LAST_PORTFOLIO_DATE, "strategy": "LW_MAXSHARPE"})
    assert resp.status_code == 200
    assert isinstance(resp.json()["evaluation"]["turnover"], float)


@_skip_no_db
def test_portfolio_equal_weight_turnover_is_approximately_zero_after_first_date(client):
    resp = client.get("/api/portfolios", params={"formation_date": EXPECTED_LAST_PORTFOLIO_DATE, "strategy": "EQUAL_WEIGHT"})
    assert resp.status_code == 200
    turnover = resp.json()["evaluation"]["turnover"]
    assert turnover is not None
    assert abs(turnover) < 1e-6


@_skip_no_db
def test_portfolio_excluded_prediction_only_date_returns_404(client):
    resp = client.get("/api/portfolios", params={"formation_date": "2022-02-25", "strategy": "LW_MAXSHARPE"})
    assert resp.status_code == 404


@_skip_no_db
def test_portfolio_unknown_future_date_returns_404(client):
    resp = client.get("/api/portfolios", params={"formation_date": "2099-01-01", "strategy": "LW_MAXSHARPE"})
    assert resp.status_code == 404


@_skip_no_db
def test_portfolio_invalid_strategy_returns_400(client):
    resp = client.get("/api/portfolios", params={"formation_date": EXPECTED_LAST_PORTFOLIO_DATE, "strategy": "NOT_A_STRATEGY"})
    assert resp.status_code == 400


@_skip_no_db
def test_portfolio_malformed_date_returns_422(client):
    resp = client.get("/api/portfolios", params={"formation_date": "not-a-date"})
    assert resp.status_code == 422


@_skip_no_db
def test_frontier_excluded_prediction_only_date_returns_404(client):
    resp = client.get("/api/frontier", params={"formation_date": "2022-02-25", "covariance": "LW"})
    assert resp.status_code == 404


@_skip_no_db
def test_frontier_unknown_future_date_returns_404(client):
    resp = client.get("/api/frontier", params={"formation_date": "2099-01-01"})
    assert resp.status_code == 404


@_skip_no_db
def test_frontier_invalid_covariance_returns_400(client):
    resp = client.get("/api/frontier", params={"formation_date": EXPECTED_LAST_PORTFOLIO_DATE, "covariance": "BAD"})
    assert resp.status_code == 400


@_skip_no_db
def test_frontier_causal_prices_never_include_post_formation_dates(client):
    """§20 causality — real behavioral proof: the exact SQL cutoff this
    route uses never returns a row dated after `formation_date`, against
    the real `prices_raw` table (through 2026-02-27)."""
    conn = db.get_connection()
    try:
        formation_date = pd.Timestamp(EXPECTED_LAST_PORTFOLIO_DATE)
        prices = frontier._load_causal_prices(conn, formation_date.date())
        assert len(prices) > 0
        assert prices["date"].max() <= formation_date
    finally:
        conn.close()


@_skip_no_db
def test_frontier_never_calls_stage_b_realized_return_helpers(client, monkeypatch):
    """§20 causality — behavioral trap: if the reconstruction path ever
    called a Stage-B (realized-return) helper, this would raise. It must
    complete cleanly, proving the call graph genuinely never reaches
    Stage B (not merely that this module happens not to import it)."""
    import optimizer.walkforward as wf

    def _boom(*args, **kwargs):
        raise AssertionError("frontier reconstruction must never call a Stage-B realized-return helper")

    monkeypatch.setattr(wf, "realized_stock_returns", _boom)
    monkeypatch.setattr(wf, "realized_portfolio_return", _boom)

    conn = db.get_connection()
    try:
        mu_arr, sigma_by_est, rf_21 = frontier.reconstruct_mu_sigma_rf(conn, EXPECTED_LAST_PORTFOLIO_DATE)
        assert np.isfinite(mu_arr).all()
        assert np.isfinite(rf_21)
    finally:
        conn.close()


# Real Phase 7 NUMERIC column precision (schema.sql): target_return/
# portfolio_vol are NUMERIC(8,6) (storage rounding up to 5e-7), sharpe_ratio
# is NUMERIC(8,4) (storage rounding up to 5e-5) — tolerances below are set
# just above those storage-precision bounds, never loosened to paper over a
# genuine reconstruction mismatch.
_RECONCILE_TOL_RETURN = 1e-5
_RECONCILE_TOL_VOL = 1e-5
_RECONCILE_TOL_SHARPE = 1e-3


@_skip_no_db
@pytest.mark.parametrize(
    "formation_date_key",
    ["first", "middle", "last"],
)
def test_frontier_reconciles_against_official_persisted_minvol_maxsharpe(client, formation_date_key):
    """§6/§21 — the mandatory reconciliation check: recompute expected
    return / volatility / Sharpe from THIS reconstructed mu_21/Sigma_21/
    rf_21 and the OFFICIAL persisted Min-Vol/Max-Sharpe weights, and
    compare against the OFFICIAL persisted target_return/portfolio_vol/
    sharpe_ratio. A material mismatch here would mean the frontier's own
    axes disagree with the frozen Phase 7 result — never something to
    paper over by loosening tolerance."""
    dates_resp = client.get("/api/portfolios/dates")
    dates = dates_resp.json()
    formation_date = {"first": dates[0], "middle": dates[len(dates) // 2], "last": dates[-1]}[formation_date_key]

    conn = db.get_connection()
    try:
        mu_arr, sigma_by_est, rf_21 = frontier.reconstruct_mu_sigma_rf(conn, formation_date)

        max_d_ret = max_d_vol = max_d_sharpe = 0.0
        for estimator in ("SAMPLE", "LW"):
            sigma_arr = sigma_by_est[estimator]
            for strategy in (f"{estimator}_MINVOL", f"{estimator}_MAXSHARPE"):
                run_id = build_run_id("p7bv1", strategy, formation_date)
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT ticker, weight, target_return, portfolio_vol, sharpe_ratio "
                        "FROM portfolios WHERE run_id = %s ORDER BY ticker",
                        (run_id,),
                    )
                    rows = cur.fetchall()
                assert len(rows) == 50
                w = pd.Series({r[0]: float(r[1]) for r in rows}).reindex(config.TICKER_UNIVERSE).to_numpy()
                official_ret, official_vol, official_sharpe = float(rows[0][2]), float(rows[0][3]), float(rows[0][4])

                recon_ret = portfolio_expected_return(w, mu_arr)
                recon_vol = portfolio_volatility(w, sigma_arr)
                recon_sharpe = portfolio_sharpe(w, mu_arr, sigma_arr, rf_21)

                max_d_ret = max(max_d_ret, abs(recon_ret - official_ret))
                max_d_vol = max(max_d_vol, abs(recon_vol - official_vol))
                max_d_sharpe = max(max_d_sharpe, abs(recon_sharpe - official_sharpe))

        assert max_d_ret < _RECONCILE_TOL_RETURN, f"expected-return reconciliation drift: {max_d_ret}"
        assert max_d_vol < _RECONCILE_TOL_VOL, f"volatility reconciliation drift: {max_d_vol}"
        assert max_d_sharpe < _RECONCILE_TOL_SHARPE, f"sharpe reconciliation drift: {max_d_sharpe}"
    finally:
        conn.close()


@_skip_no_db
@pytest.mark.parametrize("covariance", ["LW", "SAMPLE"])
def test_frontier_end_to_end_latest_date_produces_valid_dense_curve(client, covariance):
    """Full end-to-end (real DB, real SLSQP sweep) smoke test — deliberately
    run only at the latest date, once per estimator (this path costs
    roughly 15-20s per call; see the Phase 8C Efficient Frontier report's
    measured runtime). Confirms: a reasonably dense feasible curve, every
    point finite with non-negative volatility, and official markers
    present with correct provenance — never a fabricated/interpolated
    point for a target the solver could not reach."""
    resp = client.get("/api/frontier", params={"formation_date": EXPECTED_LAST_PORTFOLIO_DATE, "covariance": covariance})
    assert resp.status_code == 200
    body = resp.json()
    assert body["formation_date"] == EXPECTED_LAST_PORTFOLIO_DATE
    assert body["covariance_estimator"] == ("Ledoit-Wolf" if covariance == "LW" else "Sample")

    points = body["points"]
    assert len(points) >= 30  # "reasonably dense" per the task brief's 30-60 guidance
    vols = [p["volatility_21"] for p in points]
    rets = [p["expected_return_21"] for p in points]
    assert all(v >= 0 for v in vols)
    assert all(np.isfinite(v) and np.isfinite(r) for v, r in zip(vols, rets))

    min_vol_marker = body["markers"]["min_vol"]
    max_sharpe_marker = body["markers"]["max_sharpe"]
    equal_weight_marker = body["markers"]["equal_weight"]
    assert min_vol_marker["provenance"] == "official_phase7_persisted"
    assert max_sharpe_marker["provenance"] == "official_phase7_persisted"
    assert equal_weight_marker["provenance"] == "reconstructed_benchmark"

    # The official Min-Vol marker's volatility should sit at or very near
    # the low end of the reconstructed curve's volatility range — the same
    # numerical tolerance class as the dedicated reconciliation test, never
    # an exact-equality assumption (the curve is a discrete grid, official
    # Min-Vol need not land exactly on a grid point).
    assert min_vol_marker["volatility_21"] <= min(vols) + 5e-3


# ---------------------------------------------------------------------------
# Risk Analytics (Phase 8D-1) — formation-time component risk contribution.
# Reuses `frontier.reconstruct_mu_sigma_rf` for covariance (no second
# implementation), so these tests focus on: the volatility reconciliation
# (the core integrity test for this slice), the RC/risk-share/sector
# identities on REAL data, causality, and strategy/covariance provenance.
# ---------------------------------------------------------------------------
@_skip_no_db
def test_risk_excluded_prediction_only_date_returns_404(client):
    resp = client.get("/api/risk", params={"formation_date": "2022-02-25", "strategy": "LW_MAXSHARPE"})
    assert resp.status_code == 404


@_skip_no_db
def test_risk_unknown_future_date_returns_404(client):
    resp = client.get("/api/risk", params={"formation_date": "2099-01-01"})
    assert resp.status_code == 404


@_skip_no_db
def test_risk_invalid_strategy_returns_400(client):
    resp = client.get("/api/risk", params={"formation_date": EXPECTED_LAST_PORTFOLIO_DATE, "strategy": "NOT_A_STRATEGY"})
    assert resp.status_code == 400


@_skip_no_db
def test_risk_mismatched_covariance_for_optimized_strategy_returns_400(client):
    resp = client.get(
        "/api/risk",
        params={"formation_date": EXPECTED_LAST_PORTFOLIO_DATE, "strategy": "SAMPLE_MINVOL", "covariance": "LW"},
    )
    assert resp.status_code == 400


@_skip_no_db
def test_risk_never_calls_stage_b_realized_return_helpers(client, monkeypatch):
    """Behavioral causality trap (same pattern as the Frontier causality
    test): if Risk Analytics' reconstruction path ever called a Stage-B
    (realized-return) helper, this would raise. It must return 200,
    proving the call graph genuinely never reaches Stage B."""
    import optimizer.walkforward as wf

    def _boom(*args, **kwargs):
        raise AssertionError("risk reconstruction must never call a Stage-B realized-return helper")

    monkeypatch.setattr(wf, "realized_stock_returns", _boom)
    monkeypatch.setattr(wf, "realized_portfolio_return", _boom)

    resp = client.get("/api/risk", params={"formation_date": EXPECTED_LAST_PORTFOLIO_DATE, "strategy": "LW_MAXSHARPE"})
    assert resp.status_code == 200


@_skip_no_db
@pytest.mark.parametrize(
    "formation_date_key,strategy",
    [
        ("first", "SAMPLE_MINVOL"),
        ("first", "LW_MAXSHARPE"),
        ("middle", "LW_MINVOL"),
        ("middle", "SAMPLE_MAXSHARPE"),
        ("last", "SAMPLE_MINVOL"),
        ("last", "LW_MAXSHARPE"),
    ],
)
def test_risk_reconstructed_volatility_reconciles_with_official_persisted_portfolio_vol(client, formation_date_key, strategy):
    """§5 — the core integrity test for this slice: recompute
    sigma_p = sqrt(w^T Sigma_21 w) from the official persisted weights and
    THIS route's reconstructed Sigma_21, and compare to the official
    persisted `portfolios.portfolio_vol`. Tolerance is set just above
    `portfolio_vol`'s NUMERIC(8,6) DB storage-precision bound (schema.sql),
    matching the same justified tolerance the Frontier reconciliation test
    uses for the identical comparison."""
    dates_resp = client.get("/api/portfolios/dates")
    dates = dates_resp.json()
    formation_date = {"first": dates[0], "middle": dates[len(dates) // 2], "last": dates[-1]}[formation_date_key]

    resp = client.get("/api/risk", params={"formation_date": formation_date, "strategy": strategy})
    assert resp.status_code == 200
    reconstructed_vol = resp.json()["portfolio"]["predicted_volatility_21"]

    run_id = build_run_id("p7bv1", strategy, formation_date)
    conn = db.get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT portfolio_vol FROM portfolios WHERE run_id = %s LIMIT 1", (run_id,))
            official_vol = float(cur.fetchone()[0])
    finally:
        conn.close()

    assert abs(reconstructed_vol - official_vol) < 1e-5, (
        f"{formation_date} {strategy}: reconstructed={reconstructed_vol} official={official_vol}"
    )


@_skip_no_db
@pytest.mark.parametrize("strategy", ["LW_MAXSHARPE", "SAMPLE_MINVOL", "EQUAL_WEIGHT"])
def test_risk_identities_hold_on_real_official_portfolios(client, strategy):
    """RC sum, risk-share sum, and sector-aggregation identities, verified
    against real official Phase 7 data (not just synthetic fixtures)."""
    resp = client.get("/api/risk", params={"formation_date": EXPECTED_LAST_PORTFOLIO_DATE, "strategy": strategy})
    assert resp.status_code == 200
    body = resp.json()

    assert len(body["assets"]) == 50
    tickers = [a["ticker"] for a in body["assets"]]
    assert set(tickers) == set(config.TICKER_UNIVERSE)
    weight_sum = sum(a["weight"] for a in body["assets"])
    assert abs(weight_sum - 1.0) < 1e-4

    sigma_p = body["portfolio"]["predicted_volatility_21"]
    assert sigma_p > 0
    assert all(np.isfinite(a["component_risk_contribution"]) and np.isfinite(a["risk_share"]) for a in body["assets"])

    rc_sum = sum(a["component_risk_contribution"] for a in body["assets"])
    assert abs(rc_sum - sigma_p) < 1e-4

    share_sum = sum(a["risk_share"] for a in body["assets"])
    assert abs(share_sum - 1.0) < 1e-4

    assert len(body["sectors"]) == 2
    assert {s["sector"] for s in body["sectors"]} == {"Information Technology", "Financials"}
    sector_rc_sum = sum(s["component_risk_contribution"] for s in body["sectors"])
    assert abs(sector_rc_sum - sigma_p) < 1e-4
    sector_weight_sum = sum(s["weight"] for s in body["sectors"])
    assert abs(sector_weight_sum - 1.0) < 1e-4

    if strategy == "EQUAL_WEIGHT":
        assert body["portfolio"]["max_weight_constraint"] is None
        assert all(abs(a["weight"] - 1.0 / 50) < 1e-9 for a in body["assets"])
    else:
        assert body["portfolio"]["max_weight_constraint"] == 0.10
        assert all(a["weight"] <= 0.10 + 1e-6 for a in body["assets"])


@_skip_no_db
def test_risk_strategy_determines_covariance_provenance_truthfully(client):
    for strategy, expected_estimator in [
        ("SAMPLE_MINVOL", "Sample"),
        ("SAMPLE_MAXSHARPE", "Sample"),
        ("LW_MINVOL", "Ledoit-Wolf"),
        ("LW_MAXSHARPE", "Ledoit-Wolf"),
    ]:
        resp = client.get("/api/risk", params={"formation_date": EXPECTED_LAST_PORTFOLIO_DATE, "strategy": strategy})
        assert resp.status_code == 200
        assert resp.json()["covariance_estimator"] == expected_estimator


@_skip_no_db
def test_risk_equal_weight_covariance_selector_changes_result_without_reoptimizing_weights(client):
    resp_lw = client.get("/api/risk", params={"formation_date": EXPECTED_LAST_PORTFOLIO_DATE, "strategy": "EQUAL_WEIGHT", "covariance": "LW"})
    resp_sample = client.get("/api/risk", params={"formation_date": EXPECTED_LAST_PORTFOLIO_DATE, "strategy": "EQUAL_WEIGHT", "covariance": "SAMPLE"})
    assert resp_lw.status_code == 200 and resp_sample.status_code == 200
    body_lw, body_sample = resp_lw.json(), resp_sample.json()

    # Weights are the fixed conceptual 1/50 benchmark regardless of the
    # covariance chosen for risk ANALYSIS — never reoptimized.
    weights_lw = {a["ticker"]: a["weight"] for a in body_lw["assets"]}
    weights_sample = {a["ticker"]: a["weight"] for a in body_sample["assets"]}
    assert weights_lw == weights_sample
    assert all(abs(w - 1.0 / 50) < 1e-9 for w in weights_lw.values())

    # But the resulting predicted volatility differs (real historical
    # Sample vs. Ledoit-Wolf covariances are not identical).
    assert body_lw["portfolio"]["predicted_volatility_21"] != body_sample["portfolio"]["predicted_volatility_21"]


# ---------------------------------------------------------------------------
# Historical Evidence (Phase 8D-2) — the frozen, official Phase 7
# walk-forward experiment. `EXPECTED_BACKTEST_SUMMARY`/`EXPECTED_SPXT_SUMMARY`
# are the reconciliation targets from the frozen Phase 7 report (task
# brief §6) — never hard-coded as the response itself, only as an
# independent check on it. Tolerances are set at 1e-3 (0.1 percentage
# point / 0.001 turnover), comfortably above the report's own last
# displayed digit of rounding.
# ---------------------------------------------------------------------------
_TOL = 1e-3

EXPECTED_BACKTEST_SUMMARY = {
    "SAMPLE_MINVOL": dict(
        mean=0.0089, std=0.0389, median=0.0138, min=-0.0935, max=0.0979, hit=0.609,
        mean_to=0.0837, median_to=0.0759, max_to=0.1736, avg_maxw=0.1000, max_obs_w=0.1000, cum=0.454,
    ),
    "SAMPLE_MAXSHARPE": dict(
        mean=0.0200, std=0.0588, median=0.0243, min=-0.1187, max=0.1490, hit=0.674,
        mean_to=0.6362, median_to=0.6495, max_to=0.9000, avg_maxw=0.1000, max_obs_w=0.1000, cum=1.306,
    ),
    "LW_MINVOL": dict(
        mean=0.0090, std=0.0388, median=0.0142, min=-0.0937, max=0.0979, hit=0.609,
        mean_to=0.0806, median_to=0.0751, max_to=0.1748, avg_maxw=0.1000, max_obs_w=0.1000, cum=0.459,
    ),
    "LW_MAXSHARPE": dict(
        mean=0.0200, std=0.0588, median=0.0238, min=-0.1187, max=0.1490, hit=0.674,
        mean_to=0.6365, median_to=0.6668, max_to=0.9000, avg_maxw=0.1000, max_obs_w=0.1000, cum=1.303,
    ),
    "EQUAL_WEIGHT": dict(
        mean=0.0157, std=0.0545, median=0.0216, min=-0.1187, max=0.1454, hit=0.609,
        mean_to=0.0, median_to=0.0, max_to=0.0, avg_maxw=0.0200, max_obs_w=0.0200, cum=0.916,
    ),
}
EXPECTED_SPXT_SUMMARY = dict(mean=0.0111, std=0.0418, median=0.0149, min=-0.0998, max=0.1081, hit=0.674, cum=0.600)


@_skip_no_db
def test_backtest_experiment_shape_matches_frozen_phase7_calendar(client):
    resp = client.get("/api/backtest")
    assert resp.status_code == 200
    body = resp.json()
    exp = body["experiment"]
    assert exp["period_count"] == EXPECTED_PORTFOLIO_DATE_COUNT
    assert exp["first_formation_date"] == EXPECTED_FIRST_PORTFOLIO_DATE
    assert exp["last_formation_date"] == EXPECTED_LAST_PORTFOLIO_DATE
    assert exp["horizon_sessions"] == 21
    assert exp["covariance_window_sessions"] == 252
    assert exp["benchmark"] == "SPXT"


@_skip_no_db
def test_backtest_returns_exactly_six_series_all_with_46_real_periods(client):
    resp = client.get("/api/backtest")
    body = resp.json()
    keys = {s["key"] for s in body["series"]}
    assert keys == set(EXPECTED_STRATEGIES) | {"SPXT"}
    for s in body["series"]:
        assert len(s["periods"]) == EXPECTED_PORTFOLIO_DATE_COUNT
        formation_dates = [p["formation_date"] for p in s["periods"]]
        assert formation_dates == sorted(formation_dates)
        assert len(set(formation_dates)) == EXPECTED_PORTFOLIO_DATE_COUNT  # no duplicates/fabricated dates
        assert formation_dates[0] == EXPECTED_FIRST_PORTFOLIO_DATE
        assert formation_dates[-1] == EXPECTED_LAST_PORTFOLIO_DATE


@_skip_no_db
@pytest.mark.parametrize("strategy", EXPECTED_STRATEGIES)
def test_backtest_portfolio_summary_reconciles_with_frozen_phase7_report(client, strategy):
    """§15 mandatory reconciliation — every summary statistic independently
    recomputed by this endpoint against the frozen Phase 7 report."""
    resp = client.get("/api/backtest")
    series = next(s for s in resp.json()["series"] if s["key"] == strategy)
    summ = series["summary"]
    expected = EXPECTED_BACKTEST_SUMMARY[strategy]

    assert abs(summ["mean_return_21"] - expected["mean"]) < _TOL
    assert abs(summ["std_return_21"] - expected["std"]) < _TOL
    assert abs(summ["median_return_21"] - expected["median"]) < _TOL
    assert abs(summ["min_return_21"] - expected["min"]) < _TOL
    assert abs(summ["max_return_21"] - expected["max"]) < _TOL
    assert abs(summ["positive_period_rate"] - expected["hit"]) < _TOL
    assert abs(summ["cumulative_return"] - expected["cum"]) < 2e-3

    assert abs(summ["mean_turnover"] - expected["mean_to"]) < _TOL
    assert abs(summ["median_turnover"] - expected["median_to"]) < _TOL
    assert abs(summ["max_turnover"] - expected["max_to"]) < _TOL
    assert abs(summ["avg_max_weight"] - expected["avg_maxw"]) < _TOL
    assert abs(summ["max_observed_weight"] - expected["max_obs_w"]) < _TOL


@_skip_no_db
def test_backtest_spxt_summary_reconciles_with_frozen_phase7_report(client):
    resp = client.get("/api/backtest")
    spxt = next(s for s in resp.json()["series"] if s["key"] == "SPXT")
    summ = spxt["summary"]
    expected = EXPECTED_SPXT_SUMMARY

    assert abs(summ["mean_return_21"] - expected["mean"]) < _TOL
    assert abs(summ["std_return_21"] - expected["std"]) < _TOL
    assert abs(summ["median_return_21"] - expected["median"]) < _TOL
    assert abs(summ["min_return_21"] - expected["min"]) < _TOL
    assert abs(summ["max_return_21"] - expected["max"]) < _TOL
    assert abs(summ["positive_period_rate"] - expected["hit"]) < _TOL
    assert abs(summ["cumulative_return"] - expected["cum"]) < 2e-3
    assert summ["mean_turnover"] is None
    assert summ["avg_max_weight"] is None


@_skip_no_db
@pytest.mark.parametrize("strategy", EXPECTED_STRATEGIES)
def test_backtest_compounding_identity_never_sums_returns(client, strategy):
    """§16 compounding tests — growth begins at 1.0 (implicit, verified via
    the first period), each successive point equals
    previous_growth * (1 + period_return), the final growth matches
    `cumulative_return`, and the result is materially different from
    (and therefore not) the naive sum of period returns."""
    resp = client.get("/api/backtest")
    series = next(s for s in resp.json()["series"] if s["key"] == strategy)
    periods = series["periods"]
    returns = [p["realized_return_21"] for p in periods]
    growth = [p["growth_of_one"] for p in periods]

    level = 1.0
    for i, r in enumerate(returns):
        level *= 1.0 + r
        assert abs(growth[i] - level) < 1e-9, f"{strategy} period {i}: growth diverges from sequential compounding"

    assert abs((growth[-1] - 1.0) - series["summary"]["cumulative_return"]) < 1e-9
    assert abs(series["summary"]["cumulative_return"] - sum(returns)) > 0.01  # unambiguously not a sum


@_skip_no_db
def test_backtest_first_period_turnover_null_for_every_portfolio_strategy(client):
    resp = client.get("/api/backtest")
    for s in resp.json()["series"]:
        if s["kind"] != "portfolio":
            continue
        assert s["periods"][0]["turnover"] is None, f"{s['key']}: first-period turnover must be undefined, not 0"
        # Every later period has a real, non-null turnover.
        assert all(p["turnover"] is not None for p in s["periods"][1:])


@_skip_no_db
def test_backtest_equal_weight_later_turnover_is_truthfully_zero(client):
    resp = client.get("/api/backtest")
    eq = next(s for s in resp.json()["series"] if s["key"] == "EQUAL_WEIGHT")
    later_turnovers = [p["turnover"] for p in eq["periods"][1:]]
    assert all(abs(t) < 1e-9 for t in later_turnovers)


@_skip_no_db
def test_backtest_spxt_has_no_fabricated_turnover_or_weight_concentration(client):
    resp = client.get("/api/backtest")
    spxt = next(s for s in resp.json()["series"] if s["key"] == "SPXT")
    assert all(p["turnover"] is None and p["max_weight_observed"] is None for p in spxt["periods"])
    assert spxt["max_weight_constraint"] is None
    assert spxt["is_optimized"] is None
    assert spxt["covariance_estimator"] is None


@_skip_no_db
def test_backtest_spxt_periods_use_the_exact_official_formation_calendar(client):
    """§17 SPXT exact-date verification: SPXT's 46 (formation_date,
    target_date) pairs must be exactly the official Phase 7 portfolio
    calendar — never a nearest-date/interpolated/independently-derived
    set of dates."""
    resp = client.get("/api/backtest")
    body = resp.json()
    spxt = next(s for s in body["series"] if s["key"] == "SPXT")
    portfolio_series = next(s for s in body["series"] if s["key"] == "LW_MAXSHARPE")

    spxt_dates = [(p["formation_date"], p["target_date"]) for p in spxt["periods"]]
    portfolio_dates = [(p["formation_date"], p["target_date"]) for p in portfolio_series["periods"]]
    assert spxt_dates == portfolio_dates


@_skip_no_db
def test_backtest_endpoint_runtime_is_materially_faster_than_frontier(client):
    """§23 — this endpoint reads already-persisted figures plus one exact
    benchmark lookup; no SLSQP sweep, no ML inference. Loosely bounded
    well under Frontier's observed ~15-20s."""
    import time

    t0 = time.time()
    resp = client.get("/api/backtest")
    elapsed = time.time() - t0
    assert resp.status_code == 200
    assert elapsed < 10.0, f"/api/backtest took {elapsed:.2f}s — investigate before accepting this runtime"


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
    client.get("/api/portfolios/dates")
    client.get("/api/portfolios/strategies")
    client.get("/api/portfolios")
    client.get("/api/portfolios", params={"formation_date": "2022-03-28", "strategy": "EQUAL_WEIGHT"})
    # Deliberately just one full (real SLSQP sweep) frontier call — this
    # path costs ~15-20s; the dedicated frontier tests above already cover
    # both estimators and multiple dates.
    client.get("/api/frontier", params={"formation_date": "2022-03-28", "covariance": "LW"})
    client.get("/api/risk", params={"formation_date": "2022-03-28", "strategy": "LW_MAXSHARPE"})
    client.get("/api/risk", params={"formation_date": "2022-03-28", "strategy": "EQUAL_WEIGHT", "covariance": "SAMPLE"})
    client.get("/api/backtest")

    conn = db.get_connection()
    try:
        for table, before_count in before.items():
            after_count = db.fetch_scalar(conn, f"SELECT COUNT(*) FROM {table}")
            assert after_count == before_count, (
                f"{table} row count changed across Phase 2A API use ({before_count} -> {after_count})"
            )
    finally:
        conn.close()
