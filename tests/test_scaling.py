"""
RiskFecta Phase 5A — pooled sequence-scaler tests (pipeline/scaling.py).

Synthetic only. Covers task brief §20-M (scaler leakage) and §20-S (future
perturbation cannot change an earlier fit).
"""
from __future__ import annotations

import numpy as np
import pytest
from sklearn.preprocessing import StandardScaler

from pipeline.scaling import SequenceScaler, fit_sequence_scaler, transform_sequences


def _make_sequences(n, seq_len, n_features, loc=0.0, scale=1.0, seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(loc=loc, scale=scale, size=(n, seq_len, n_features))


# ---------------------------------------------------------------------------
# Basic fit/transform correctness
# ---------------------------------------------------------------------------
def test_fit_matches_manual_standard_scaler_on_pooled_rows():
    train = _make_sequences(20, 10, 3, loc=5.0, scale=2.0, seed=1)
    scaler = fit_sequence_scaler(train)

    manual = StandardScaler().fit(train.reshape(-1, 3))
    np.testing.assert_allclose(scaler.mean_, manual.mean_)
    np.testing.assert_allclose(scaler.scale_, manual.scale_)
    assert scaler.n_features == 3


def test_transform_produces_pooled_zero_mean_unit_variance():
    train = _make_sequences(50, 10, 4, loc=10.0, scale=3.0, seed=2)
    scaler = fit_sequence_scaler(train)
    transformed = transform_sequences(train, scaler)
    flat = transformed.reshape(-1, 4)
    np.testing.assert_allclose(flat.mean(axis=0), np.zeros(4), atol=1e-8)
    np.testing.assert_allclose(flat.std(axis=0), np.ones(4), atol=1e-8)


def test_fit_rejects_wrong_ndim_or_empty():
    with pytest.raises(ValueError):
        fit_sequence_scaler(np.zeros((10, 5)))  # 2D, not 3D
    with pytest.raises(ValueError):
        fit_sequence_scaler(np.zeros((0, 60, 5)))  # zero samples


def test_transform_rejects_feature_count_mismatch():
    train = _make_sequences(10, 10, 3, seed=3)
    scaler = fit_sequence_scaler(train)
    wrong = _make_sequences(5, 10, 4, seed=4)
    with pytest.raises(ValueError):
        transform_sequences(wrong, scaler)


# ---------------------------------------------------------------------------
# M. Scaler leakage — fit only ever sees what is passed in.
# ---------------------------------------------------------------------------
def test_scaler_fit_on_training_slice_ignores_a_companion_oos_array():
    train = _make_sequences(30, 10, 3, loc=0.0, scale=1.0, seed=5)
    oos_normal = _make_sequences(10, 10, 3, loc=0.0, scale=1.0, seed=6)
    oos_extreme = oos_normal.copy()
    oos_extreme[:] = 1e9  # wildly different distribution

    scaler_with_normal_oos_elsewhere = fit_sequence_scaler(train)
    scaler_with_extreme_oos_elsewhere = fit_sequence_scaler(train)

    # The scaler is fit from `train` alone in both cases — an OOS array
    # existing "elsewhere" in the caller's scope (never passed to
    # fit_sequence_scaler) cannot influence it, however extreme its values.
    np.testing.assert_array_equal(scaler_with_normal_oos_elsewhere.mean_, scaler_with_extreme_oos_elsewhere.mean_)
    np.testing.assert_array_equal(scaler_with_normal_oos_elsewhere.scale_, scaler_with_extreme_oos_elsewhere.scale_)


# ---------------------------------------------------------------------------
# S. Future perturbation — mutating the OOS half of a shared underlying
# array must not retroactively change an already-computed fit on the
# train-only slice (aliasing safety), nor change a fresh re-fit on the
# same (unmutated) train slice.
# ---------------------------------------------------------------------------
def test_mutating_oos_slice_of_shared_array_does_not_change_train_fit():
    full = _make_sequences(40, 10, 3, loc=2.0, scale=1.5, seed=7)
    n_train = 25
    train_view = full[:n_train]
    oos_view = full[n_train:]

    scaler_before = fit_sequence_scaler(train_view)

    oos_view[:] = 1e12  # mutate only the OOS slice of the shared array

    scaler_after = fit_sequence_scaler(full[:n_train])  # re-fit the SAME train slice
    np.testing.assert_allclose(scaler_before.mean_, scaler_after.mean_)
    np.testing.assert_allclose(scaler_before.scale_, scaler_after.scale_)


def test_extreme_future_values_never_enter_transform_of_earlier_data():
    train = _make_sequences(20, 10, 3, loc=0.0, scale=1.0, seed=8)
    scaler = fit_sequence_scaler(train)
    transformed_before = transform_sequences(train, scaler)

    # A separate, extreme "future" sequence array transformed with the SAME
    # already-fit scaler must not alter the scaler's own parameters, nor
    # the transform of the original training sequences.
    future_extreme = np.full((5, 10, 3), 1e9)
    _ = transform_sequences(future_extreme, scaler)

    transformed_after = transform_sequences(train, scaler)
    np.testing.assert_array_equal(transformed_before, transformed_after)
    assert isinstance(scaler, SequenceScaler)
