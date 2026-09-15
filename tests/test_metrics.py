"""
RiskFecta Phase 4A — evaluation metrics tests (models/metrics.py).

Covers task brief §14, §17-K: exact known-value checks and NaN/constant-
vector degenerate-case handling.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from models.metrics import (
    compute_all_metrics,
    directional_accuracy,
    mae,
    pearson_corr,
    rmse,
    spearman_corr,
)


def test_mae_exact_known_value():
    y_true = [1.0, 2.0, 3.0]
    y_pred = [1.5, 1.5, 3.5]
    assert mae(y_true, y_pred) == pytest.approx((0.5 + 0.5 + 0.5) / 3)


def test_rmse_exact_known_value():
    y_true = [0.0, 0.0]
    y_pred = [3.0, 4.0]
    assert rmse(y_true, y_pred) == pytest.approx(math.sqrt((9 + 16) / 2))


def test_directional_accuracy_exact_known_value_excludes_zero_actuals():
    y_true = [0.01, -0.02, 0.0, 0.03]
    y_pred = [0.02, -0.01, 0.5, -0.01]  # signs: +,-,(excluded),-
    # matches: index0 (+/+) yes, index1 (-/-) yes, index3 (+/-) no -> 2/3
    assert directional_accuracy(y_true, y_pred) == pytest.approx(2 / 3)


def test_directional_accuracy_all_zero_actuals_is_nan_not_zero_or_one():
    assert math.isnan(directional_accuracy([0.0, 0.0], [0.1, -0.1]))


def test_pearson_corr_exact_known_value_perfect_correlation():
    y_true = [1.0, 2.0, 3.0, 4.0]
    y_pred = [2.0, 4.0, 6.0, 8.0]
    assert pearson_corr(y_true, y_pred) == pytest.approx(1.0)


def test_pearson_corr_constant_vector_is_nan_not_zero():
    y_true = [1.0, 1.0, 1.0]
    y_pred = [0.1, 0.2, 0.3]
    assert math.isnan(pearson_corr(y_true, y_pred))


def test_pearson_corr_insufficient_observations_is_nan():
    assert math.isnan(pearson_corr([1.0], [2.0]))
    assert math.isnan(pearson_corr([], []))


def test_spearman_corr_exact_known_value_monotonic():
    y_true = [1.0, 2.0, 3.0, 4.0]
    y_pred = [10.0, 20.0, 30.0, 5.0]  # not perfectly monotonic
    # Spearman on ranks [1,2,3,4] vs ranks [2,3,4,1]: known value via formula
    from scipy.stats import spearmanr
    expected, _ = spearmanr(y_true, y_pred)
    assert spearman_corr(y_true, y_pred) == pytest.approx(expected)


def test_spearman_corr_constant_vector_is_nan():
    assert math.isnan(spearman_corr([1.0, 1.0, 1.0], [0.1, 0.5, 0.2]))


def test_nan_predictions_dropped_pairwise_not_fabricated():
    y_true = [1.0, 2.0, 3.0]
    y_pred = [1.0, float("nan"), 3.0]
    # Only the two finite pairs should count.
    assert mae(y_true, y_pred) == pytest.approx(0.0)


def test_mismatched_shapes_raises():
    with pytest.raises(ValueError):
        mae([1.0, 2.0], [1.0])


def test_compute_all_metrics_reports_n_obs_after_dropping_nans():
    y_true = [1.0, 2.0, float("nan")]
    y_pred = [1.0, 2.0, 3.0]
    out = compute_all_metrics(y_true, y_pred)
    assert out["n_obs"] == 2
    assert set(out.keys()) == {"mae", "rmse", "directional_accuracy", "pearson_corr", "spearman_corr", "n_obs"}


def test_empty_input_all_metrics_nan_not_error():
    out = compute_all_metrics([], [])
    assert out["n_obs"] == 0
    for k in ("mae", "rmse", "directional_accuracy", "pearson_corr", "spearman_corr"):
        assert math.isnan(out[k])
