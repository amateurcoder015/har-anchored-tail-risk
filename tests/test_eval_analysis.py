import numpy as np
import pandas as pd
import pytest

from volgate.evaluate.analysis import expiry_did, hac_ols, state_regression


def test_hac_ols_recovers_coefficients_and_drops_constant_columns():
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"x": rng.normal(size=2000), "flat": 1.0})
    y = 0.5 + 2.0 * X["x"] + rng.normal(0, 0.1, 2000)
    t = hac_ols(y, X)
    assert "flat" not in t.index
    assert t.loc["x", "coef"] == pytest.approx(2.0, abs=0.02)
    assert t.loc["const", "coef"] == pytest.approx(0.5, abs=0.02)


def test_expiry_did_picks_up_interaction():
    idx = pd.bdate_range("2025-01-01", "2026-06-30")
    e = pd.Series((idx.dayofweek == 1).astype(float), index=idx)
    post = idx >= pd.Timestamp("2025-09-01")
    rng = np.random.default_rng(1)
    d = pd.Series(rng.normal(0, 0.1, len(idx)) - 0.5 * e.to_numpy() * post, index=idx)
    t = expiry_did(d, e)
    assert t.loc["expiry_x_post", "coef"] == pytest.approx(-0.5, abs=0.1)
    assert t.loc["expiry_x_post", "p"] < 0.01


def test_state_regression_standardizes():
    idx = pd.bdate_range("2023-01-02", periods=500)
    rng = np.random.default_rng(2)
    s = pd.DataFrame({"vix": rng.normal(15, 5, 500)}, index=idx)
    d = pd.Series(0.1 * (s["vix"] - 15) / 5 + rng.normal(0, 0.05, 500), index=idx)
    t = state_regression(d, s)
    assert t.loc["vix", "coef"] == pytest.approx(0.1, abs=0.02)
