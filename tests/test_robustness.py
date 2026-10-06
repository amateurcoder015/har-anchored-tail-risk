import numpy as np
import pandas as pd
import pytest
from scipy import stats

from hart.robustness import acerbi_szekely_z2, power_projection, sign_counts

A = 0.025


def test_z2_near_zero_when_correct_and_negative_when_es_too_shallow():
    rng = np.random.default_rng(0)
    y = rng.standard_normal(200_000)
    q = stats.norm.ppf(A)
    es = -stats.norm.pdf(q) / A
    z_good = acerbi_szekely_z2(y, np.full(y.size, q), np.full(y.size, es), A)
    z_bad = acerbi_szekely_z2(y, np.full(y.size, q), np.full(y.size, es * 0.7), A)
    assert abs(z_good) < 0.05 and z_bad < -0.3


def test_power_projection_independent_assets():
    rng = np.random.default_rng(1)
    T, N = 900, 20
    D = pd.DataFrame(rng.normal(-0.02, 1.0, (T, N)))
    p = power_projection(D)
    assert abs(p["rho"]) < 0.05
    # with rho ~ 0 the projection reduces to mean / sqrt(s2 / (T * N))
    assert p["t_at_N"] == pytest.approx(D.to_numpy().mean() / np.sqrt(1.0 / (T * N)), rel=0.1)
    def t_of(n):
        return p["mean_diff"] / np.sqrt(p["s2"] * (p["rho"] + (1 - p["rho"]) / n) / T)
    k = p["n_required"]
    assert abs(t_of(k)) >= 2.487 and (k == 1 or abs(t_of(k - 1)) < 2.487)  # smallest such N


def test_power_projection_correlated_assets_has_ceiling():
    rng = np.random.default_rng(2)
    T, N = 900, 20
    common = rng.normal(0, 1.0, (T, 1))
    D = pd.DataFrame(-0.001 + 0.9 * common + 0.44 * rng.normal(0, 1.0, (T, N)))
    p = power_projection(D)
    assert p["rho"] > 0.6 and p["n_required"] is None


def test_sign_counts():
    per = {"a": pd.DataFrame({"x": [1.0, 1.0], "y": [2.0, 2.0]}),
           "b": pd.DataFrame({"x": [3.0, 3.0], "y": [2.0, 2.0]})}
    assert sign_counts(per, "x", "y") == (1, 2)


def test_z2_critical_value_shrinks_with_sample_length():
    from hart.robustness import z2_critical
    c250 = z2_critical(250, A, reps=3000, seed=0)
    c920 = z2_critical(920, A, reps=3000, seed=0)
    assert -0.85 < c250 < -0.55  # Acerbi-Szekely report about -0.70 for T = 250
    assert -0.45 < c920 < -0.30 and c920 > c250


def test_power_projection_is_signed_and_calibrated():
    rng = np.random.default_rng(3)
    T, N = 900, 20
    D = pd.DataFrame(+0.05 + rng.normal(0, 1.0, (T, N)))  # first method WORSE
    p = power_projection(D)
    assert p["n_required"] is None  # never counts a significant result in the wrong direction
