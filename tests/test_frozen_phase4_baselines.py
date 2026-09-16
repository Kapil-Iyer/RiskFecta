"""Regression tests for the frozen Phase 4A baseline artifact
(models/frozen_phase4_baselines.py). These values are not derived from the
database — this test suite exists precisely so any accidental edit to the
literals is caught immediately, since nothing else in the pipeline can
re-derive or cross-check them against a live source."""
from __future__ import annotations

from models.frozen_phase4_baselines import FROZEN_BASELINE_RESULTS

EXPECTED = {
    "historical_mean": dict(mae=0.07312, rmse=0.09722, directional_accuracy=0.5366, pearson_corr=0.02854, spearman_corr=0.01673),
    "momentum_3m": dict(mae=0.14254, rmse=0.18803, directional_accuracy=0.5077, pearson_corr=-0.00132, spearman_corr=-0.03109),
    "ridge": dict(mae=0.07831, rmse=0.10421, directional_accuracy=0.4761, pearson_corr=0.00158, spearman_corr=-0.03153),
}


def test_exactly_three_frozen_baselines():
    assert {b.model for b in FROZEN_BASELINE_RESULTS} == set(EXPECTED.keys())
    assert len(FROZEN_BASELINE_RESULTS) == 3


def test_frozen_baseline_values_match_authoritative_phase4a_results():
    by_model = {b.model: b for b in FROZEN_BASELINE_RESULTS}
    for model, expected in EXPECTED.items():
        actual = by_model[model]
        for field, value in expected.items():
            assert getattr(actual, field) == value, f"{model}.{field} drifted from the frozen result"


def test_frozen_baselines_share_the_experiment_observation_count():
    # Same 47-date x 50-ticker walk-forward window as XGBoost/LSTM/Ensemble
    # (independently confirmed live in tests/test_api_db_integration.py).
    for b in FROZEN_BASELINE_RESULTS:
        assert b.n_obs == 2350
