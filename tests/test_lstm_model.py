"""
RiskFecta Phase 5A — PooledLSTM architecture/training tests (models/lstm.py).

Synthetic-only (task brief §2, §20-O/P/Q, §21): every fixture here is a
tiny, fabricated dataset. Nothing in this file touches real Bloomberg
features, and no real LSTM OOS performance is computed or inspected.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import nn

import config
from models.lstm import (
    DEFAULT_SEED,
    PooledLSTM,
    LSTM_FEATURE_COLS,
    SEQUENCE_LENGTH,
    assert_no_forbidden_architecture,
    predict_lstm,
    set_seed,
    train_lstm,
)


def _synthetic_dataset(n, seq_len=8, n_features=3, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, seq_len, n_features)).astype(np.float32)
    # y depends only on the LAST timestep's first feature + small noise —
    # a trivially learnable synthetic regression target.
    y = (0.5 * X[:, -1, 0] + 0.01 * rng.normal(size=n)).astype(np.float32)
    return X, y


# ---------------------------------------------------------------------------
# Feature surface / config wiring
# ---------------------------------------------------------------------------
def test_feature_surface_and_sequence_length_match_config():
    assert LSTM_FEATURE_COLS == list(config.LSTM_ALL_FEATURES)
    assert SEQUENCE_LENGTH == config.LSTM_SEQ


# ---------------------------------------------------------------------------
# O. Model shape
# ---------------------------------------------------------------------------
def test_forward_shape_is_one_scalar_per_sequence():
    model = PooledLSTM(input_size=5, hidden_size=8, num_layers=1)
    x = torch.zeros(4, 60, 5)
    out = model(x)
    assert out.shape == (4,)
    assert torch.isfinite(out).all()


def test_forward_handles_arbitrary_batch_and_seq_len():
    model = PooledLSTM(input_size=3, hidden_size=4, num_layers=1)
    x = torch.randn(7, 25, 3)
    out = model(x)
    assert out.shape == (7,)


# ---------------------------------------------------------------------------
# Architecture lock (task brief §13)
# ---------------------------------------------------------------------------
def test_default_model_has_no_forbidden_components():
    model = PooledLSTM()
    assert_no_forbidden_architecture(model)  # must not raise
    assert not any(isinstance(m, nn.Embedding) for m in model.modules())
    assert not any(isinstance(m, nn.GRU) for m in model.modules())
    assert not model.lstm.bidirectional


def test_forbidden_component_is_rejected():
    class BadModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.gru = nn.GRU(input_size=3, hidden_size=4)

    with pytest.raises(ValueError):
        assert_no_forbidden_architecture(BadModel())


# ---------------------------------------------------------------------------
# P. Training smoke test — tiny synthetic dataset only.
# ---------------------------------------------------------------------------
def test_train_lstm_smoke_test_loss_finite_and_params_update():
    X_train, y_train = _synthetic_dataset(40, seq_len=8, n_features=3, seed=1)
    X_val, y_val = _synthetic_dataset(10, seq_len=8, n_features=3, seed=2)

    set_seed(DEFAULT_SEED)
    reference = PooledLSTM(input_size=3, hidden_size=6, num_layers=1)
    reference_head_weight = reference.head.weight.detach().clone()

    result = train_lstm(
        X_train, y_train, X_val, y_val,
        hidden_size=6, num_layers=1, max_epochs=5, batch_size=8, patience=5, seed=DEFAULT_SEED,
    )

    assert len(result.train_losses) > 0
    assert len(result.val_losses) == len(result.train_losses)
    assert all(np.isfinite(v) for v in result.train_losses)
    assert all(np.isfinite(v) for v in result.val_losses)

    # Parameters actually moved away from a freshly-initialized (same-seed)
    # reference model — proves backprop/optimizer step ran, not just that
    # the loop executed.
    assert not torch.allclose(result.model.head.weight.detach(), reference_head_weight)

    preds = predict_lstm(result.model, X_val)
    assert preds.shape == (len(X_val),)
    assert np.isfinite(preds).all()


def test_train_lstm_rejects_empty_training_set():
    X_train = np.zeros((0, 8, 3), dtype=np.float32)
    y_train = np.zeros((0,), dtype=np.float32)
    X_val, y_val = _synthetic_dataset(5, seq_len=8, n_features=3, seed=3)
    with pytest.raises(ValueError):
        train_lstm(X_train, y_train, X_val, y_val, max_epochs=2)


def test_train_lstm_early_stopping_never_exceeds_max_epochs():
    X_train, y_train = _synthetic_dataset(30, seq_len=6, n_features=2, seed=4)
    X_val, y_val = _synthetic_dataset(8, seq_len=6, n_features=2, seed=5)
    result = train_lstm(
        X_train, y_train, X_val, y_val,
        hidden_size=4, max_epochs=6, batch_size=8, patience=2, seed=1,
    )
    assert len(result.train_losses) <= 6


# ---------------------------------------------------------------------------
# Q. Reproducibility
# ---------------------------------------------------------------------------
def test_train_lstm_is_reproducible_given_fixed_seed():
    X_train, y_train = _synthetic_dataset(30, seq_len=6, n_features=2, seed=9)
    X_val, y_val = _synthetic_dataset(8, seq_len=6, n_features=2, seed=10)

    result1 = train_lstm(X_train, y_train, X_val, y_val, hidden_size=4, max_epochs=4, batch_size=8, seed=123)
    result2 = train_lstm(X_train, y_train, X_val, y_val, hidden_size=4, max_epochs=4, batch_size=8, seed=123)

    np.testing.assert_allclose(result1.train_losses, result2.train_losses, rtol=1e-5)
    np.testing.assert_allclose(result1.val_losses, result2.val_losses, rtol=1e-5)

    preds1 = predict_lstm(result1.model, X_val)
    preds2 = predict_lstm(result2.model, X_val)
    np.testing.assert_allclose(preds1, preds2, rtol=1e-5)
