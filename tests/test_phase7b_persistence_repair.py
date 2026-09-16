"""
RiskFecta Phase 7B — persistence repair tests (turnover-row omission +
atomic transaction). scripts/run_phase7b_official.py. Task brief
"IMPLEMENTATION TESTS" items 1-12.

Root cause being fixed: schema.sql's `risk_metrics.metric_value` is
NOT NULL, but the locked turnover methodology requires each strategy's
first Phase 7 formation to have an undefined (NaN) turnover. The fix
OMITS that one row entirely (never a NULL, never a sentinel) and makes
`portfolios`+`risk_metrics` persistence atomic (one transaction) so a
failure on either insert can never again leave a half-persisted
official experiment.

Mock-only: NO real database connection is opened anywhere in this file.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import optimizer.persistence as persist_mod
import optimizer.walkforward as wf

SCRIPT_PATH = Path(__file__).resolve().parent.parent / "scripts" / "run_phase7b_official.py"


def _load_driver_module():
    spec = importlib.util.spec_from_file_location("phase7b_official_driver_repair", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


driver = _load_driver_module()


# ---------------------------------------------------------------------------
# Fakes for the atomic-transaction tests (never a real DB connection).
# ---------------------------------------------------------------------------
class FakeCursor:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class FakeConn:
    def __init__(self, initial_autocommit=True):
        self.autocommit = initial_autocommit
        self.committed = False
        self.rolled_back = False
        self.autocommit_history = [initial_autocommit]

    def cursor(self):
        return FakeCursor()

    def commit(self):
        self.committed = True

    def rollback(self):
        self.rolled_back = True


TICKERS = ["AAA", "BBB", "CCC"]


def _tiny_dfs():
    run_id = persist_mod.build_run_id("p7btest", "EQUAL_WEIGHT", pd.Timestamp("2022-03-28"))
    portfolios_df = persist_mod.build_portfolio_rows(run_id, TICKERS, [1 / 3, 1 / 3, 1 / 3], 0.01, 0.05, 0.2)
    risk_metrics_df = persist_mod.build_risk_metric_rows(run_id, {"realized_return_21": 0.01, "max_weight_observed": 1 / 3})
    return portfolios_df, risk_metrics_df


# ---------------------------------------------------------------------------
# 1. First-date turnover is still NaN/undefined in-memory.
# ---------------------------------------------------------------------------
def test_first_date_turnover_is_nan_in_memory():
    tv = wf.turnover(np.array([0.5, 0.5]), None, ["A", "B"])
    assert np.isnan(tv)


# ---------------------------------------------------------------------------
# 2/3. Persistence omits exactly the first turnover row; never a sentinel.
# ---------------------------------------------------------------------------
def test_build_period_risk_metrics_omits_turnover_when_nan():
    d = driver._build_period_risk_metrics(0.02, float("nan"), 0.05)
    assert set(d.keys()) == {"realized_return_21", "max_weight_observed"}
    assert "turnover" not in d


def test_build_period_risk_metrics_includes_turnover_when_finite():
    d = driver._build_period_risk_metrics(0.02, 0.03, 0.05)
    assert set(d.keys()) == {"realized_return_21", "max_weight_observed", "turnover"}
    assert d["turnover"] == pytest.approx(0.03)


def test_build_period_risk_metrics_never_fabricates_a_sentinel():
    d = driver._build_period_risk_metrics(0.02, float("nan"), 0.05)
    for v in d.values():
        assert np.isfinite(v)
        assert v not in (0, -1, 999)


def test_build_period_risk_metrics_rejects_nonfinite_always_present_metrics():
    with pytest.raises(ValueError, match="realized_return_21"):
        driver._build_period_risk_metrics(float("nan"), 0.01, 0.05)
    with pytest.raises(ValueError, match="max_weight_observed"):
        driver._build_period_risk_metrics(0.01, 0.01, float("nan"))


# ---------------------------------------------------------------------------
# 4. Exact per-strategy metric counts: 46/46/45 across a full synthetic
# 46-date, 5-strategy experiment.
# ---------------------------------------------------------------------------
def test_full_synthetic_experiment_has_exact_46_46_45_metric_counts():
    n_dates = 46
    dates = pd.bdate_range("2022-03-28", periods=n_dates * 21, freq="B")[::21][:n_dates]
    frames = []
    for strat in wf.STRATEGIES:
        for i, fd in enumerate(dates):
            tv = float("nan") if i == 0 else 0.05
            run_id = persist_mod.build_run_id("p7btest", strat, fd)
            metrics = driver._build_period_risk_metrics(0.01, tv, 0.02)
            frames.append(persist_mod.build_risk_metric_rows(run_id, metrics))
    risk_metrics_df = pd.concat(frames, ignore_index=True)

    n_strategies = len(wf.STRATEGIES)
    assert int((risk_metrics_df["metric_name"] == "realized_return_21").sum()) == n_dates * n_strategies
    assert int((risk_metrics_df["metric_name"] == "max_weight_observed").sum()) == n_dates * n_strategies
    assert int((risk_metrics_df["metric_name"] == "turnover").sum()) == (n_dates - 1) * n_strategies
    assert len(risk_metrics_df) == n_dates * n_strategies * 2 + (n_dates - 1) * n_strategies
    assert not risk_metrics_df["metric_value"].isna().any()


# ---------------------------------------------------------------------------
# 5. Later valid turnover values persist normally (non-NaN never dropped).
# ---------------------------------------------------------------------------
def test_later_turnover_values_are_never_dropped():
    d = driver._build_period_risk_metrics(0.02, 0.0, 0.05)  # exactly-zero turnover, e.g. unchanged EQUAL_WEIGHT
    assert "turnover" in d
    assert d["turnover"] == 0.0


# ---------------------------------------------------------------------------
# 6/9. Atomic transaction: one commit, both inserts, on success.
# ---------------------------------------------------------------------------
def test_persist_official_experiment_commits_once_on_success(monkeypatch):
    portfolios_df, risk_metrics_df = _tiny_dfs()
    calls = []

    def fake_execute_values(cur, sql, records, page_size=1000):
        calls.append(sql)

    monkeypatch.setattr(driver.persist_mod, "execute_values", fake_execute_values)
    conn = FakeConn(initial_autocommit=True)

    n_port, n_risk = driver._persist_official_experiment(conn, portfolios_df, risk_metrics_df)

    assert len(calls) == 2
    assert any("portfolios" in c for c in calls)
    assert any("risk_metrics" in c for c in calls)
    assert conn.committed is True
    assert conn.rolled_back is False
    assert conn.autocommit is True  # restored to prior value afterward
    assert n_port == len(portfolios_df)
    assert n_risk == len(risk_metrics_df)


def test_persist_official_experiment_disables_autocommit_during_the_call(monkeypatch):
    portfolios_df, risk_metrics_df = _tiny_dfs()
    seen_autocommit_during_insert = []

    def fake_execute_values(cur, sql, records, page_size=1000):
        seen_autocommit_during_insert.append(conn.autocommit)

    monkeypatch.setattr(driver.persist_mod, "execute_values", fake_execute_values)
    conn = FakeConn(initial_autocommit=True)
    driver._persist_official_experiment(conn, portfolios_df, risk_metrics_df)

    assert all(ac is False for ac in seen_autocommit_during_insert)  # off for both inserts
    assert conn.autocommit is True  # restored after


# ---------------------------------------------------------------------------
# 7. Simulated risk_metrics failure rolls back portfolios too.
# ---------------------------------------------------------------------------
def test_risk_metrics_failure_rolls_back_portfolios_insert(monkeypatch):
    portfolios_df, risk_metrics_df = _tiny_dfs()
    calls = []

    def fake_execute_values(cur, sql, records, page_size=1000):
        calls.append(sql)
        if "risk_metrics" in sql:
            raise RuntimeError("simulated NOT NULL violation")

    monkeypatch.setattr(driver.persist_mod, "execute_values", fake_execute_values)
    conn = FakeConn()

    with pytest.raises(RuntimeError, match="simulated NOT NULL violation"):
        driver._persist_official_experiment(conn, portfolios_df, risk_metrics_df)

    assert len(calls) == 2  # portfolios was attempted, then risk_metrics failed
    assert conn.committed is False
    assert conn.rolled_back is True
    assert conn.autocommit is True  # still restored even on failure


# ---------------------------------------------------------------------------
# 8. Simulated portfolios failure leaves both tables unchanged (risk_metrics
# insert never even attempted).
# ---------------------------------------------------------------------------
def test_portfolios_failure_never_attempts_risk_metrics_insert(monkeypatch):
    portfolios_df, risk_metrics_df = _tiny_dfs()
    calls = []

    def fake_execute_values(cur, sql, records, page_size=1000):
        calls.append(sql)
        if "portfolios" in sql:
            raise RuntimeError("simulated portfolios failure")

    monkeypatch.setattr(driver.persist_mod, "execute_values", fake_execute_values)
    conn = FakeConn()

    with pytest.raises(RuntimeError, match="simulated portfolios failure"):
        driver._persist_official_experiment(conn, portfolios_df, risk_metrics_df)

    assert len(calls) == 1  # risk_metrics insert never reached
    assert not any("risk_metrics" in c for c in calls)
    assert conn.committed is False
    assert conn.rolled_back is True


# ---------------------------------------------------------------------------
# 10. Existing duplicate/collision protection remains effective (re-confirm
# here; full coverage lives in test_optimizer_persistence_run_identity.py).
# ---------------------------------------------------------------------------
def test_collision_protection_still_raises_before_any_insert():
    class _ExistingConn:
        def cursor(self):
            class _Cur:
                def __enter__(self):
                    return self

                def __exit__(self, *a):
                    return False

                def execute(self, sql, params):
                    self._found = True

                def fetchone(self):
                    return (1,)

            return _Cur()

    with pytest.raises(ValueError, match="already exists"):
        persist_mod.assert_run_id_available(_ExistingConn(), "portfolios", "p7btest_EQUAL_WEIGHT_2022-03-28")


# ---------------------------------------------------------------------------
# 11. Validation and aggregation still occur before the transaction begins
# (source-order + no conn.commit/rollback anywhere but the persist function).
# ---------------------------------------------------------------------------
def test_commit_and_rollback_calls_exist_only_inside_persist_official_experiment():
    import inspect
    module_src = inspect.getsource(driver)
    persist_src = inspect.getsource(driver._persist_official_experiment)
    for call in ("conn.commit(", "conn.rollback("):
        total = module_src.count(call)
        within = persist_src.count(call)
        assert total >= 1
        assert total == within


def test_validate_and_aggregate_precede_persist_in_main_source():
    import inspect
    main_src = inspect.getsource(driver.main)
    aggregate_pos = main_src.index("_compute_all_aggregates(")
    validate_pos = main_src.index("_validate_complete_experiment(")
    persist_pos = main_src.index("_persist_official_experiment(")
    assert aggregate_pos < validate_pos < persist_pos
