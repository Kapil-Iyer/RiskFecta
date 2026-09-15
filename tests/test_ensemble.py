"""
RiskFecta Phase 6A — ensemble arithmetic + diagnostics tests
(models/ensemble.py). Synthetic/adversarial only — no real database
connection, no model training/invocation anywhere in this file (task
brief §13, §19-O).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from models.ensemble import (
    CorrelationResult,
    DirectionalAgreementResult,
    DisagreementBreakdown,
    assert_expected_shape,
    assert_identity_integrity,
    build_ensemble_frame,
    compare_models,
    compute_ensemble_pred,
    directional_agreement,
    disagreement_outcome_breakdown,
    dispersion_diagnostics,
    per_date_rank_ic,
    prediction_correlation,
    rank_ic_summary,
    residual_correlation,
    top_bottom_quintile_spread,
)


def _identity_df(rows):
    """rows: list of dicts with ticker, forecast_date, xgb_pred, lstm_pred,
    optional actual_return/target_date."""
    df = pd.DataFrame(rows)
    if "target_date" not in df.columns:
        df["target_date"] = pd.NaT
    if "actual_return" not in df.columns:
        df["actual_return"] = np.nan
    return df


# ---------------------------------------------------------------------------
# A. exact 50/50 arithmetic
# ---------------------------------------------------------------------------
def test_ensemble_exact_50_50_arithmetic():
    xgb = pd.Series([0.10])
    lstm = pd.Series([0.02])
    out = compute_ensemble_pred(xgb, lstm)
    assert out.iloc[0] == pytest.approx(0.06)


# ---------------------------------------------------------------------------
# B. negative/mixed predictions — exact arithmetic preserved
# ---------------------------------------------------------------------------
def test_ensemble_negative_and_mixed_predictions_exact():
    xgb = pd.Series([-0.10, 0.05, -0.02])
    lstm = pd.Series([0.04, -0.05, -0.08])
    out = compute_ensemble_pred(xgb, lstm)
    expected = [(-0.10 + 0.04) / 2, (0.05 - 0.05) / 2, (-0.02 - 0.08) / 2]
    assert out.tolist() == pytest.approx(expected)


# ---------------------------------------------------------------------------
# C. identity alignment — ticker/date/target identity retained
# ---------------------------------------------------------------------------
def test_build_ensemble_frame_retains_identity_columns():
    df = _identity_df([
        {"ticker": "AAPL", "forecast_date": pd.Timestamp("2022-02-25"),
         "target_date": pd.Timestamp("2022-03-28"), "xgb_pred": 0.01, "lstm_pred": 0.02, "actual_return": 0.015},
        {"ticker": "MSFT", "forecast_date": pd.Timestamp("2022-02-25"),
         "target_date": pd.Timestamp("2022-03-28"), "xgb_pred": -0.01, "lstm_pred": 0.03, "actual_return": -0.005},
    ])
    out = build_ensemble_frame(df)
    assert list(out["ticker"]) == ["AAPL", "MSFT"]
    assert list(out["forecast_date"]) == [pd.Timestamp("2022-02-25")] * 2
    assert list(out["target_date"]) == [pd.Timestamp("2022-03-28")] * 2
    assert out.set_index("ticker")["ensemble_pred"].to_dict() == pytest.approx({"AAPL": 0.015, "MSFT": 0.01})


# ---------------------------------------------------------------------------
# D. missing model prediction — fail loudly, never silently drop/average
# ---------------------------------------------------------------------------
def test_assert_identity_integrity_raises_on_missing_xgb_pred():
    df = _identity_df([
        {"ticker": "AAPL", "forecast_date": pd.Timestamp("2022-02-25"), "xgb_pred": np.nan, "lstm_pred": 0.02},
    ])
    with pytest.raises(ValueError, match="missing xgb_pred"):
        assert_identity_integrity(df)


def test_assert_identity_integrity_raises_on_missing_lstm_pred():
    df = _identity_df([
        {"ticker": "AAPL", "forecast_date": pd.Timestamp("2022-02-25"), "xgb_pred": 0.01, "lstm_pred": np.nan},
    ])
    with pytest.raises(ValueError, match="missing lstm_pred"):
        assert_identity_integrity(df)


def test_build_ensemble_frame_raises_rather_than_dropping_missing_row():
    df = _identity_df([
        {"ticker": "AAPL", "forecast_date": pd.Timestamp("2022-02-25"), "xgb_pred": 0.01, "lstm_pred": np.nan},
        {"ticker": "MSFT", "forecast_date": pd.Timestamp("2022-02-25"), "xgb_pred": 0.02, "lstm_pred": 0.03},
    ])
    with pytest.raises(ValueError):
        build_ensemble_frame(df)


# ---------------------------------------------------------------------------
# E. duplicate identity — rejected
# ---------------------------------------------------------------------------
def test_assert_identity_integrity_raises_on_duplicate_identity():
    df = _identity_df([
        {"ticker": "AAPL", "forecast_date": pd.Timestamp("2022-02-25"), "xgb_pred": 0.01, "lstm_pred": 0.02},
        {"ticker": "AAPL", "forecast_date": pd.Timestamp("2022-02-25"), "xgb_pred": 0.015, "lstm_pred": 0.025},
    ])
    with pytest.raises(ValueError, match="duplicate"):
        assert_identity_integrity(df)


def test_assert_expected_shape_raises_on_wrong_row_count():
    df = _identity_df([
        {"ticker": "AAPL", "forecast_date": pd.Timestamp("2022-02-25"), "xgb_pred": 0.01, "lstm_pred": 0.02},
    ])
    with pytest.raises(ValueError):
        assert_expected_shape(df, n_tickers=50, n_dates=47)


# ---------------------------------------------------------------------------
# F. no clipping/transformation — extreme finite values average exactly
# ---------------------------------------------------------------------------
def test_ensemble_no_clipping_on_extreme_values():
    xgb = pd.Series([100.0, -100.0])
    lstm = pd.Series([50.0, -50.0])
    out = compute_ensemble_pred(xgb, lstm)
    assert out.tolist() == pytest.approx([75.0, -75.0])


# ---------------------------------------------------------------------------
# G. existing metrics reused correctly
# ---------------------------------------------------------------------------
def test_compare_models_reuses_metrics_module_exactly():
    df = _identity_df([
        {"ticker": "AAPL", "forecast_date": pd.Timestamp("2022-02-25"), "xgb_pred": 0.02, "lstm_pred": 0.01, "actual_return": 0.03},
        {"ticker": "MSFT", "forecast_date": pd.Timestamp("2022-02-25"), "xgb_pred": -0.01, "lstm_pred": -0.02, "actual_return": -0.015},
    ])
    df = build_ensemble_frame(df)
    out = compare_models(df)
    assert set(out.index) == {"xgb_pred", "lstm_pred", "ensemble_pred"}
    assert {"mae", "rmse", "directional_accuracy", "pearson_corr", "spearman_corr", "n_obs"} <= set(out.columns)


def test_compare_models_excludes_missing_columns_not_invents_them():
    df = _identity_df([
        {"ticker": "AAPL", "forecast_date": pd.Timestamp("2022-02-25"), "xgb_pred": 0.02, "lstm_pred": 0.01, "actual_return": 0.03},
    ])
    df = build_ensemble_frame(df)
    out = compare_models(df, model_cols=["xgb_pred", "historical_mean_pred"])
    assert list(out.index) == ["xgb_pred"]


# ---------------------------------------------------------------------------
# H. prediction-correlation diagnostics
# ---------------------------------------------------------------------------
def test_prediction_correlation_perfect_agreement():
    xgb = [0.01, 0.02, 0.03, 0.04]
    lstm = [0.02, 0.04, 0.06, 0.08]  # perfectly linearly related
    out = prediction_correlation(xgb, lstm)
    assert isinstance(out, CorrelationResult)
    assert out.pearson == pytest.approx(1.0)
    assert out.spearman == pytest.approx(1.0)


def test_prediction_correlation_constant_vector_is_nan():
    xgb = [0.01, 0.01, 0.01]
    lstm = [0.02, 0.03, 0.01]
    out = prediction_correlation(xgb, lstm)
    assert np.isnan(out.pearson)
    assert np.isnan(out.spearman)


# ---------------------------------------------------------------------------
# I. residual-correlation diagnostics
# ---------------------------------------------------------------------------
def test_residual_correlation_computed_from_prediction_minus_actual():
    # Integer-valued inputs so the residual differences are exactly
    # constant (no floating-point subtraction noise).
    xgb = [1.0, 2.0, 3.0]
    lstm = [1.5, 2.5, 3.5]
    actual = [0.0, 1.0, 2.0]
    out = residual_correlation(xgb, lstm, actual)
    # residuals: xgb=[1,1,1], lstm=[1.5,1.5,1.5] -> both constant -> NaN (undefined)
    assert np.isnan(out.pearson)


def test_residual_correlation_varying_residuals():
    xgb = [0.05, 0.20, 0.02]
    lstm = [0.06, 0.25, -0.01]
    actual = [0.04, 0.10, 0.05]
    out = residual_correlation(xgb, lstm, actual)
    assert not np.isnan(out.pearson)


# ---------------------------------------------------------------------------
# J. directional agreement/disagreement logic including zero cases
# ---------------------------------------------------------------------------
def test_directional_agreement_basic_counts():
    xgb = [0.01, -0.02, 0.03, -0.04]
    lstm = [0.02, -0.01, -0.03, -0.05]  # agree, agree, disagree, agree
    out = directional_agreement(xgb, lstm)
    assert isinstance(out, DirectionalAgreementResult)
    assert out.n_total == 4
    assert out.n_zero_sign == 0
    assert out.n_comparable == 4
    assert out.n_agree == 3
    assert out.n_disagree == 1
    assert out.agreement_rate == pytest.approx(0.75)


def test_directional_agreement_zero_sign_excluded_explicitly():
    xgb = [0.0, 0.01, -0.02]
    lstm = [0.05, 0.0, -0.03]
    out = directional_agreement(xgb, lstm)
    assert out.n_total == 3
    assert out.n_zero_sign == 2  # first row (xgb=0) and second row (lstm=0)
    assert out.n_comparable == 1
    assert out.n_agree == 1  # third row: both negative
    assert out.agreement_rate == pytest.approx(1.0)


def test_directional_agreement_all_zero_sign_yields_nan_rate():
    xgb = [0.0]
    lstm = [0.0]
    out = directional_agreement(xgb, lstm)
    assert out.n_comparable == 0
    assert np.isnan(out.agreement_rate)


def test_disagreement_outcome_breakdown_xgb_and_lstm_correct_split():
    xgb = [0.05, -0.05]
    lstm = [-0.05, 0.05]
    actual = [0.02, 0.02]  # row1: xgb sign matches -> xgb correct; row2: lstm sign matches -> lstm correct
    out = disagreement_outcome_breakdown(xgb, lstm, actual)
    assert isinstance(out, DisagreementBreakdown)
    assert out.n_disagree == 2
    assert out.xgb_correct_lstm_wrong == 1
    assert out.lstm_correct_xgb_wrong == 1
    assert out.actual_zero_excluded == 0


def test_disagreement_outcome_breakdown_zero_actual_excluded():
    xgb = [0.05]
    lstm = [-0.05]
    actual = [0.0]
    out = disagreement_outcome_breakdown(xgb, lstm, actual)
    assert out.n_disagree == 1
    assert out.xgb_correct_lstm_wrong == 0
    assert out.lstm_correct_xgb_wrong == 0
    assert out.actual_zero_excluded == 1


def test_disagreement_outcome_breakdown_ignores_agreeing_rows():
    xgb = [0.05, 0.03]
    lstm = [0.06, -0.02]  # row1 agrees (both positive), row2 disagrees
    actual = [0.01, 0.01]
    out = disagreement_outcome_breakdown(xgb, lstm, actual)
    assert out.n_disagree == 1


# ---------------------------------------------------------------------------
# dispersion diagnostics
# ---------------------------------------------------------------------------
def test_dispersion_diagnostics_shapes():
    xgb = [0.01, 0.02, 0.03]
    lstm = [0.02, 0.02, 0.02]
    ens = compute_ensemble_pred(pd.Series(xgb), pd.Series(lstm)).tolist()
    out = dispersion_diagnostics(xgb, lstm, ens)
    assert out.var_lstm == pytest.approx(0.0)
    assert out.mean_abs_diff_xgb_lstm == pytest.approx(np.mean([0.01, 0.0, 0.01]))


# ---------------------------------------------------------------------------
# K. cross-sectional per-date rank calculation
# ---------------------------------------------------------------------------
def test_per_date_rank_ic_perfect_ranking():
    rows = []
    for t in ["AAPL", "MSFT", "NVDA", "AMD"]:
        pass
    df = pd.DataFrame({
        "forecast_date": [pd.Timestamp("2022-02-25")] * 4,
        "ticker": ["AAPL", "MSFT", "NVDA", "AMD"],
        "ensemble_pred": [0.01, 0.02, 0.03, 0.04],
        "actual_return": [0.001, 0.002, 0.003, 0.004],
    })
    out = per_date_rank_ic(df, pred_col="ensemble_pred")
    assert len(out) == 1
    assert out.loc[0, "rank_ic"] == pytest.approx(1.0)
    assert out.loc[0, "n"] == 4


def test_per_date_rank_ic_multiple_dates_independent():
    df = pd.DataFrame({
        "forecast_date": [pd.Timestamp("2022-02-25")] * 3 + [pd.Timestamp("2022-03-25")] * 3,
        "ticker": ["AAPL", "MSFT", "NVDA"] * 2,
        "ensemble_pred": [0.01, 0.02, 0.03, 0.03, 0.02, 0.01],
        "actual_return": [0.001, 0.002, 0.003, 0.001, 0.002, 0.003],
    })
    out = per_date_rank_ic(df, pred_col="ensemble_pred")
    assert len(out) == 2
    assert out.loc[0, "rank_ic"] == pytest.approx(1.0)
    assert out.loc[1, "rank_ic"] == pytest.approx(-1.0)


def test_rank_ic_summary_distribution():
    rank_ic_df = pd.DataFrame({"forecast_date": [1, 2, 3], "rank_ic": [0.5, -0.2, 0.3], "n": [50, 50, 50]})
    out = rank_ic_summary(rank_ic_df)
    assert out["n_dates"] == 3
    assert out["mean"] == pytest.approx((0.5 - 0.2 + 0.3) / 3)
    assert out["frac_positive"] == pytest.approx(2 / 3)


def test_top_bottom_quintile_spread_basic():
    tickers = [f"T{i}" for i in range(20)]
    preds = list(range(20))  # T19 has highest pred, T0 lowest
    actuals = [float(i) / 100 for i in range(20)]
    df = pd.DataFrame({
        "forecast_date": [pd.Timestamp("2022-02-25")] * 20,
        "ticker": tickers,
        "pred": preds,
        "actual_return": actuals,
    })
    out = top_bottom_quintile_spread(df, pred_col="pred", group_size=5)
    assert len(out) == 1
    row = out.iloc[0]
    # top 5 preds are T15..T19 (actuals .15-.19), bottom 5 are T0..T4 (.00-.04)
    assert row["top_mean_return"] == pytest.approx(np.mean([0.15, 0.16, 0.17, 0.18, 0.19]))
    assert row["bottom_mean_return"] == pytest.approx(np.mean([0.00, 0.01, 0.02, 0.03, 0.04]))
    assert row["spread"] == pytest.approx(row["top_mean_return"] - row["bottom_mean_return"])


def test_top_bottom_quintile_spread_insufficient_names_is_nan():
    df = pd.DataFrame({
        "forecast_date": [pd.Timestamp("2022-02-25")] * 3,
        "ticker": ["A", "B", "C"],
        "pred": [1, 2, 3],
        "actual_return": [0.01, 0.02, 0.03],
    })
    out = top_bottom_quintile_spread(df, pred_col="pred", group_size=10)
    assert np.isnan(out.loc[0, "spread"])
