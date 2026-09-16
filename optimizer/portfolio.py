"""
RiskFecta Phase 7A — constrained mean-variance optimization
(optimizer/portfolio.py).

NEW PHASE 7 PLANNER LOCKS (see optimizer/covariance.py for the covariance
half of the methodology write-up):

  LOCK C — the optimizer works entirely in 21-trading-session units:
    mu_21    = persisted `ensemble_pred` (ML_SPEC.md §8/§19/§24) — UNCHANGED,
               never recomputed/rescaled by this module.
    Sigma_21 = 21 * Sigma_session (optimizer.covariance.scale_covariance_to_horizon)
    rf_21    = ((USGG10YR/100)/252) * 21  (`rf_horizon` below)

  LOCK D — MAX_WEIGHT = config.MAX_WEIGHT = 0.10 (ML_SPEC.md §26 decision
    gate, resolved by planner fiat — NOT evidence-derived; no max-weight
    sensitivity analysis is authorized in Phase 7A).

Causality: every function in this module is a pure function of its
arguments — none of them touch the database, the global session calendar,
or "current" anything. A caller cannot make these functions see a future
realized return because none of them accept realized/future data as an
input in the first place; the causality guarantee lives entirely in how
`mu`/`sigma`/`rf` were constructed upstream (optimizer/covariance.py,
`predictions.ensemble_pred` as of T).
"""
from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import minimize

import config

DEFAULT_HORIZON = config.FORECAST_HORIZON


# ---------------------------------------------------------------------------
# Risk-free rate — 21-session horizon lock
# ---------------------------------------------------------------------------
def rf_horizon(usgg10yr: float, horizon: int = None) -> float:
    """rf_21 = ((USGG10YR/100)/252) * horizon (default
    `config.FORECAST_HORIZON` = 21). Builds on the already-frozen
    ML_SPEC.md §25 session conversion `(USGG10YR/100)/252` — never
    `USGG10YR/252` without the /100 percent-normalization step, and never
    the annual rf compared directly against a 21-session mu."""
    horizon = DEFAULT_HORIZON if horizon is None else horizon
    return ((usgg10yr / 100.0) / 252.0) * horizon


# ---------------------------------------------------------------------------
# Benchmarks
# ---------------------------------------------------------------------------
def equal_weight_benchmark(n: int) -> np.ndarray:
    """Exact 1/n per name (ML_SPEC.md §27) — never optimized."""
    if n <= 0:
        raise ValueError("equal_weight_benchmark: n must be positive")
    return np.full(n, 1.0 / n)


# ---------------------------------------------------------------------------
# Strict ticker / matrix alignment
# ---------------------------------------------------------------------------
def align_mu_sigma(mu: pd.Series, sigma: pd.DataFrame, tickers: Sequence[str]):
    """Strict ticker/matrix alignment — never relies on implicit
    pandas join/reindex ordering. Returns (mu_array, sigma_array) in
    EXACTLY `tickers` order. Fails loudly on:
      - missing ticker (in `tickers`, absent from mu or sigma)
      - extra/unexpected ticker (present in mu/sigma but not requested)
      - duplicate ticker identity in mu's index or sigma's index/columns
      - NaN/Inf in mu or sigma
    """
    tickers = list(tickers)
    if len(set(tickers)) != len(tickers):
        raise ValueError(f"align_mu_sigma: duplicate tickers requested: {tickers}")

    mu_index = list(mu.index)
    if len(set(mu_index)) != len(mu_index):
        raise ValueError("align_mu_sigma: duplicate ticker identity in mu")
    missing_mu = [t for t in tickers if t not in mu_index]
    extra_mu = [t for t in mu_index if t not in tickers]
    if missing_mu:
        raise ValueError(f"align_mu_sigma: ticker(s) missing from mu: {missing_mu}")
    if extra_mu:
        raise ValueError(f"align_mu_sigma: unexpected extra ticker(s) in mu: {extra_mu}")

    sigma_index = list(sigma.index)
    sigma_cols = list(sigma.columns)
    if len(set(sigma_index)) != len(sigma_index) or len(set(sigma_cols)) != len(sigma_cols):
        raise ValueError("align_mu_sigma: duplicate ticker identity in sigma index/columns")
    for label, axis in (("index", sigma_index), ("columns", sigma_cols)):
        missing = [t for t in tickers if t not in axis]
        extra = [t for t in axis if t not in tickers]
        if missing:
            raise ValueError(f"align_mu_sigma: ticker(s) missing from sigma {label}: {missing}")
        if extra:
            raise ValueError(f"align_mu_sigma: unexpected extra ticker(s) in sigma {label}: {extra}")

    mu_arr = mu.reindex(tickers).to_numpy(dtype=float)
    sigma_arr = sigma.reindex(index=tickers, columns=tickers).to_numpy(dtype=float)

    if not np.isfinite(mu_arr).all():
        raise ValueError("align_mu_sigma: mu contains NaN/Inf")
    if not np.isfinite(sigma_arr).all():
        raise ValueError("align_mu_sigma: sigma contains NaN/Inf")
    return mu_arr, sigma_arr


# ---------------------------------------------------------------------------
# SciPy-boundary robustness: force a C-contiguous copy of every array
# handed to `scipy.optimize.minimize` (NEW — see the Phase 7 SLSQP
# memory-layout robustness fix report). This changes ONLY memory layout,
# never numerical values (`np.ascontiguousarray` on an already-C-contiguous
# array is a no-op; on an F-contiguous array it copies the identical
# values into C order — verified empirically: max abs difference 0.0 on
# the real formation date that surfaced this). Not a numerical fix, not a
# solver-behavior change — SLSQP's own default tolerances/iterations/
# initial guess are all unchanged.
# ---------------------------------------------------------------------------
def _ensure_c_contiguous(arr) -> np.ndarray:
    return np.ascontiguousarray(arr, dtype=float)


# ---------------------------------------------------------------------------
# Constraint helpers
# ---------------------------------------------------------------------------
def _bounds(n: int, max_weight: float):
    if max_weight * n < 1.0 - 1e-9:
        raise ValueError(
            f"infeasible constraints: max_weight={max_weight} * n={n} = {max_weight * n} < 1 "
            "— fully-invested + max-weight cannot both be satisfied"
        )
    return [(0.0, max_weight) for _ in range(n)]


def _fully_invested_constraint():
    return {"type": "eq", "fun": lambda w: float(np.sum(w) - 1.0)}


def _clean_weights(w: np.ndarray, n: int, max_weight: float, tol: float = 1e-6) -> np.ndarray:
    """Validate an optimizer solution against the constraints it was
    supposed to satisfy (never trust solver success silently), then snap
    negligible numerical noise (e.g. -1e-12) to the exact bound."""
    w = np.asarray(w, dtype=float)
    if w.shape != (n,):
        raise ValueError(f"_clean_weights: expected shape ({n},), got {w.shape}")
    if not np.isfinite(w).all():
        raise ValueError("_clean_weights: optimizer produced non-finite weights")
    if w.min() < -tol or w.max() > max_weight + tol:
        raise ValueError(
            f"_clean_weights: optimizer produced out-of-bound weights "
            f"(min={w.min():.6g}, max={w.max():.6g}, max_weight={max_weight})"
        )
    if abs(w.sum() - 1.0) > tol:
        raise ValueError(f"_clean_weights: weights do not sum to 1 (sum={w.sum():.6g})")
    return np.clip(w, 0.0, max_weight)


# ---------------------------------------------------------------------------
# Portfolio statistics (pure — no lookup, no I/O)
# ---------------------------------------------------------------------------
def portfolio_expected_return(w, mu) -> float:
    return float(np.asarray(w, dtype=float) @ np.asarray(mu, dtype=float))


def portfolio_volatility(w, sigma) -> float:
    w = np.asarray(w, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    return float(np.sqrt(max(float(w @ sigma @ w), 0.0)))


def portfolio_sharpe(w, mu, sigma, rf) -> float:
    vol = portfolio_volatility(w, sigma)
    if vol <= 0:
        return float("nan")
    return float((portfolio_expected_return(w, mu) - rf) / vol)


# ---------------------------------------------------------------------------
# Minimum-volatility portfolio — mu-independent by construction (doesn't
# even accept a mu argument, so it is structurally impossible for expected
# returns to influence this objective — ML_SPEC.md §26).
# ---------------------------------------------------------------------------
def minimum_volatility_weights(sigma, max_weight: float = None) -> np.ndarray:
    # C-contiguous normalization (never changes numerical values — see
    # `_ensure_c_contiguous`'s docstring) before handing arrays to SciPy:
    # an F-contiguous Sigma (e.g. from `align_mu_sigma`'s
    # `DataFrame.reindex(...).to_numpy()`) has been observed to make
    # SLSQP's internal finite-difference line search fail nondeterministically
    # ("Positive directional derivative for linesearch") on some real
    # 50-asset inputs, even though the underlying values are identical.
    sigma = _ensure_c_contiguous(sigma)
    max_weight = config.MAX_WEIGHT if max_weight is None else max_weight
    n = sigma.shape[0]
    bounds = _bounds(n, max_weight)
    w0 = equal_weight_benchmark(n)

    def objective(w):
        return float(w @ sigma @ w)

    result = minimize(
        objective, w0, method="SLSQP", bounds=bounds,
        constraints=[_fully_invested_constraint()],
        options={"maxiter": 1000, "ftol": 1e-14},
    )
    if not result.success:
        raise ValueError(f"minimum_volatility_weights: optimizer did not converge: {result.message}")
    return _clean_weights(result.x, n, max_weight)


# ---------------------------------------------------------------------------
# Maximum-Sharpe portfolio
# ---------------------------------------------------------------------------
# NEW — deterministic one-retry SLSQP robustness rule (see the Phase 7
# deterministic Max-Sharpe SLSQP robustness fix report). SLSQP's default
# single-start, finite-difference line search occasionally fails to
# converge on real 50-asset (mu, Sigma, rf) inputs even when they are
# entirely well-formed (finite, PSD, feasible) — observed on 3 of the 46
# real Phase 7 formation dates. This is a solver-robustness limitation,
# not a methodology or data problem, and is handled with exactly ONE
# deterministic retry: same objective/mu/Sigma/rf/bounds/constraints/
# solver/tolerances, changing ONLY the initial guess from equal-weight to
# the corresponding minimum-volatility portfolio (computed from the SAME
# Sigma and constraints). The min-vol portfolio is an INITIALIZATION
# only — it is never returned in place of the Max-Sharpe solution.
_SLSQP_LINESEARCH_FAILURE = "Positive directional derivative for linesearch"


def maximum_sharpe_weights(mu, sigma, rf: float, max_weight: float = None) -> np.ndarray:
    """Maximize (w.T @ mu - rf) / sqrt(w.T @ Sigma @ w) s.t. long-only /
    fully-invested / max-weight, using mu/sigma/rf in IDENTICAL 21-session
    units (LOCK C/D).

    Primary attempt starts from equal-weight, exactly as before. If —
    and only if — that attempt fails with SciPy's known
    "Positive directional derivative for linesearch" condition, retries
    EXACTLY ONCE from the minimum-volatility portfolio as x0 (same
    Max-Sharpe objective, same mu/Sigma/rf/bounds/constraints/solver
    settings — nothing else changes). Any other failure, or a second
    failure on retry, raises immediately — never retried further."""
    mu = _ensure_c_contiguous(mu)
    sigma = _ensure_c_contiguous(sigma)
    max_weight = config.MAX_WEIGHT if max_weight is None else max_weight
    n = len(mu)
    bounds = _bounds(n, max_weight)

    def neg_sharpe(w):
        vol = float(np.sqrt(max(float(w @ sigma @ w), 1e-18)))
        return -float((w @ mu - rf) / vol)

    def _solve(w0):
        return minimize(
            neg_sharpe, w0, method="SLSQP", bounds=bounds,
            constraints=[_fully_invested_constraint()],
            options={"maxiter": 1000, "ftol": 1e-14},
        )

    result = _solve(equal_weight_benchmark(n))

    if not result.success and _SLSQP_LINESEARCH_FAILURE in (result.message or ""):
        w0_retry = minimum_volatility_weights(sigma, max_weight=max_weight)  # x0 only
        result = _solve(w0_retry)

    if not result.success:
        raise ValueError(f"maximum_sharpe_weights: optimizer did not converge: {result.message}")
    return _clean_weights(result.x, n, max_weight)


# ---------------------------------------------------------------------------
# Efficient Frontier — GENERIC machinery only. Accepts explicit
# target_return(s) supplied by the caller; no official real-experiment
# grid is chosen here (that grid was never frozen by ML_SPEC.md/
# BUILD_PLAN.md — see the Phase 7A report).
# ---------------------------------------------------------------------------
def _achievable_return_range(mu: np.ndarray, max_weight: float):
    """Max/min achievable w.T @ mu under sum(w)=1, 0<=w<=max_weight: greedily
    fill the highest-(lowest-)mu names up to max_weight each."""
    n = len(mu)

    def extreme(order):
        remaining = 1.0
        total = 0.0
        for idx in order:
            take = min(max_weight, remaining)
            total += take * mu[idx]
            remaining -= take
            if remaining <= 1e-12:
                break
        return total

    hi = extreme(np.argsort(-mu))
    lo = extreme(np.argsort(mu))
    return lo, hi


def efficient_frontier_point(mu, sigma, target_return: float, max_weight: float = None) -> np.ndarray:
    """Minimize variance s.t. sum(w)=1, w.T@mu=target_return, box
    constraints. Raises on an infeasible target_return (outside the
    achievable range given max_weight) rather than returning a
    nonsensical/clamped result."""
    mu = _ensure_c_contiguous(mu)
    sigma = _ensure_c_contiguous(sigma)
    max_weight = config.MAX_WEIGHT if max_weight is None else max_weight
    n = len(mu)
    bounds = _bounds(n, max_weight)
    w0 = equal_weight_benchmark(n)

    lo, hi = _achievable_return_range(mu, max_weight)
    if not (lo - 1e-9 <= target_return <= hi + 1e-9):
        raise ValueError(
            f"efficient_frontier_point: target_return={target_return:.6g} infeasible given "
            f"max_weight={max_weight} (achievable range [{lo:.6g}, {hi:.6g}])"
        )

    def objective(w):
        return float(w @ sigma @ w)

    constraints = [
        _fully_invested_constraint(),
        {"type": "eq", "fun": lambda w: float(w @ mu) - target_return},
    ]
    result = minimize(
        objective, w0, method="SLSQP", bounds=bounds, constraints=constraints,
        options={"maxiter": 1000, "ftol": 1e-14},
    )
    if not result.success:
        raise ValueError(f"efficient_frontier_point: optimizer did not converge: {result.message}")
    w = _clean_weights(result.x, n, max_weight)
    realized_return = portfolio_expected_return(w, mu)
    if abs(realized_return - target_return) > 1e-4:
        raise ValueError(
            f"efficient_frontier_point: solved portfolio return {realized_return:.6g} does not match "
            f"target {target_return:.6g} within tolerance"
        )
    return w


def efficient_frontier(mu, sigma, target_returns: Sequence[float], max_weight: float = None) -> List[Dict]:
    """Batch wrapper: one row per requested target_return. Does not itself
    invent a grid — `target_returns` is entirely the caller's choice. A
    single infeasible target does not abort the batch; it is reported as
    `feasible=False` with `weights=None` (so a caller sweeping a candidate
    grid can see which points are and are not achievable) rather than
    silently dropped or raising for the whole batch."""
    out = []
    for target in target_returns:
        try:
            w = efficient_frontier_point(mu, sigma, target, max_weight=max_weight)
            out.append({
                "target_return": float(target),
                "feasible": True,
                "weights": w,
                "volatility": portfolio_volatility(w, sigma),
            })
        except ValueError as exc:
            out.append({
                "target_return": float(target),
                "feasible": False,
                "weights": None,
                "volatility": float("nan"),
                "error": str(exc),
            })
    return out
