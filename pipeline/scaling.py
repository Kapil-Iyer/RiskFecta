"""
RiskFecta Phase 5A — leakage-safe pooled scaler for LSTM sequence inputs
(ML_SPEC.md §22; task brief §11).

Fits one `StandardScaler` pooled across every timestep of every training
sequence handed to it (task brief §11: "because this is a pooled model, a
pooled scaler may be appropriate ... it must remain fold-local and
train-only"). This module has no knowledge of folds, eligibility, or which
sequences are "training" vs. "OOS" — exactly like `models.xgboost_model`
trusts `pipeline.folds`'s already-purged rows, this module trusts that
whatever array the caller labels `train_sequences` has already been
filtered to purge-safe training-only sequences (`pipeline.sequences` +
the outer fold's formation cutoff). It reads nothing else, fetches nothing
from a database, and has no access to "the rest of history" — so a caller
that passes only genuine training data structurally cannot leak OOS/future
information into the fitted scaler (see tests/test_scaling.py's
future-perturbation adversarial cases).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class SequenceScaler:
    mean_: np.ndarray
    scale_: np.ndarray
    n_features: int


def fit_sequence_scaler(train_sequences: np.ndarray) -> SequenceScaler:
    """Fit a per-feature mean/std pooled over every (sample, timestep) row
    of `train_sequences` (shape `(n_samples, seq_len, n_features)` —
    every ticker's training sequences are expected to already be combined
    into this one array by the caller, task brief §7/§11 "pooled scaler").

    Fits only on what is passed in — never re-reads a wider panel, never
    consults OOS/validation/March data. Raises rather than silently
    handling degenerate input (zero training sequences), since a scaler
    fit on nothing is meaningless and must not silently become an identity
    transform.
    """
    if train_sequences.ndim != 3:
        raise ValueError(
            "fit_sequence_scaler: expected an array shaped "
            "(n_samples, seq_len, n_features), got shape "
            f"{train_sequences.shape}"
        )
    n, seq_len, n_features = train_sequences.shape
    if n == 0:
        raise ValueError("fit_sequence_scaler: train_sequences has zero samples — cannot fit a scaler")

    flat = train_sequences.reshape(-1, n_features)
    scaler = StandardScaler()
    scaler.fit(flat)
    return SequenceScaler(mean_=scaler.mean_.copy(), scale_=scaler.scale_.copy(), n_features=n_features)


def transform_sequences(sequences: np.ndarray, scaler: SequenceScaler) -> np.ndarray:
    """Apply an already-fitted `SequenceScaler` to `sequences`
    (`(n, seq_len, n_features)`) — elementwise `(x - mean) / scale`,
    broadcasting the fitted per-feature statistics across every sample and
    timestep. Used identically for training, internal-validation, and
    outer-OOS sequences (task brief §11 steps 4-6) — the SAME fitted
    `scaler` object is always the caller's responsibility to reuse; this
    function never re-fits.
    """
    if sequences.ndim != 3:
        raise ValueError(
            "transform_sequences: expected an array shaped "
            "(n_samples, seq_len, n_features), got shape "
            f"{sequences.shape}"
        )
    if sequences.shape[-1] != scaler.n_features:
        raise ValueError(
            "transform_sequences: feature-count mismatch — sequences has "
            f"{sequences.shape[-1]} features, scaler was fit on {scaler.n_features}"
        )
    return (sequences - scaler.mean_) / scaler.scale_
