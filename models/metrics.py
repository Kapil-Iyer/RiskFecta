"""
RiskFecta Phase 4A — OOS evaluation metrics (ML_SPEC.md §20).

Every function here operates only on genuine OOS (prediction, realized
target) pairs handed to it by the caller — this module never fetches data,
never knows about folds/purging, and never fabricates a value for a
degenerate case (insufficient observations, a constant vector for
correlation, or NaNs): it returns `float("nan")` explicitly rather than a
misleading number (task brief §14).
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
from scipy.stats import spearmanr


def _paired_finite(y_true: Sequence[float], y_pred: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    """Drop any pair where either side is NaN/inf — pairwise, never a
    silent full-column drop or a fabricated fill."""
    t = np.asarray(y_true, dtype=float)
    p = np.asarray(y_pred, dtype=float)
    if t.shape != p.shape:
        raise ValueError("metrics: y_true and y_pred must be the same shape")
    mask = np.isfinite(t) & np.isfinite(p)
    return t[mask], p[mask]


def mae(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    t, p = _paired_finite(y_true, y_pred)
    if len(t) == 0:
        return float("nan")
    return float(np.mean(np.abs(t - p)))


def rmse(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    t, p = _paired_finite(y_true, y_pred)
    if len(t) == 0:
        return float("nan")
    return float(np.sqrt(np.mean((t - p) ** 2)))


def directional_accuracy(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """Sign match between forecast and realized return. Observations with
    `actual_return == 0` are excluded from the denominator (ML_SPEC.md §20,
    explicit) — a genuine zero realized return has no defined sign to match
    against, so it must not be silently counted as a hit or a miss either
    way.
    """
    t, p = _paired_finite(y_true, y_pred)
    nonzero = t != 0
    t, p = t[nonzero], p[nonzero]
    if len(t) == 0:
        return float("nan")
    return float(np.mean(np.sign(t) == np.sign(p)))


def pearson_corr(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    t, p = _paired_finite(y_true, y_pred)
    if len(t) < 2:
        return float("nan")
    if np.std(t) == 0 or np.std(p) == 0:
        # Constant vector: correlation is mathematically undefined, not 0/1.
        return float("nan")
    return float(np.corrcoef(t, p)[0, 1])


def spearman_corr(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    t, p = _paired_finite(y_true, y_pred)
    if len(t) < 2:
        return float("nan")
    if np.all(t == t[0]) or np.all(p == p[0]):
        return float("nan")
    corr, _ = spearmanr(t, p)
    return float(corr) if np.isfinite(corr) else float("nan")


def compute_all_metrics(y_true: Sequence[float], y_pred: Sequence[float]) -> dict:
    """Convenience bundle of every required §20 metric for one (model, fold
    or full-history) OOS prediction set. Pure aggregation — no persistence,
    no fold/purge knowledge.
    """
    return {
        "mae": mae(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "directional_accuracy": directional_accuracy(y_true, y_pred),
        "pearson_corr": pearson_corr(y_true, y_pred),
        "spearman_corr": spearman_corr(y_true, y_pred),
        "n_obs": int(len(_paired_finite(y_true, y_pred)[0])),
    }
