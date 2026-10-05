import numpy as np
import pytest
from scipy import stats

from volgate.combine.baselines import (combine_pairs, fit_min_score, fit_previous_best,
                                       fit_relative_score, mean_combo, median_combo,
                                       rolling_combination)

A = 0.025


def _toy(n=3000, seed=0):
    """Model 0 is exact for N(0, 1); models 1, 2 are mis-scaled."""
    rng = np.random.default_rng(seed)
    y = rng.standard_normal(n)
    q = stats.norm.ppf(A)
    v0, e0 = q, -stats.norm.pdf(q) / A
    scales = np.array([1.0, 0.5, 2.0])
    V = np.tile(v0 * scales, (n, 1))
    E = np.tile(e0 * scales, (n, 1))
    return y, V, E


def test_combine_pairs_mean_equals_mean_es():
    _, V, E = _toy(10)
    v, e = mean_combo(V, E)
    np.testing.assert_allclose(e, E.mean(axis=1))
    np.testing.assert_allclose(v, V.mean(axis=1))
    v2, e2 = combine_pairs(V, E, np.array([0.2, 0.3, 0.5]), np.array([1.0, 0.0, 0.0]))
    assert np.all(e2 <= v2)


def test_median_keeps_ordering():
    _, V, E = _toy(10)
    v, e = median_combo(V, E)
    assert np.all(e <= v)


def test_min_score_finds_correct_model():
    y, V, E = _toy()
    wq, ws, _ = fit_min_score(y, V, E, A)
    assert wq.sum() == pytest.approx(1) and ws.sum() == pytest.approx(1)
    v, e = combine_pairs(V, E, wq, ws)
    assert v[0] == pytest.approx(V[0, 0], rel=0.1)
    assert e[0] == pytest.approx(E[0, 0], rel=0.1)


def test_relative_score_favours_best_and_previous_best_is_one_hot():
    y, V, E = _toy()
    w, lam = fit_relative_score(y, V, E, A)
    assert np.argmax(w) == 0 and lam > 0
    assert fit_previous_best(y, V, E, A).tolist() == [1.0, 0.0, 0.0]


def test_rolling_combination_is_causal_and_skips_nan_rows():
    y, V, E = _toy(600)
    V = V.copy()
    V[300, 1] = np.nan
    base = rolling_combination(y, V, E, A, "relative_score", refit_every=50, min_train=250)
    assert np.all(np.isnan(base[0][:250]))
    finite = np.isfinite(base[0][250:])
    assert finite.sum() == 349 and not finite[50]  # only row 300 (NaN input) is missing
    y2 = y.copy()
    y2[400] = -50.0
    pert = rolling_combination(y2, V, E, A, "relative_score", refit_every=50, min_train=250)
    np.testing.assert_array_equal(base[0][:401], pert[0][:401])
    np.testing.assert_array_equal(base[1][:401], pert[1][:401])


def test_unknown_method_raises():
    y, V, E = _toy(300)
    with pytest.raises(ValueError, match="method"):
        rolling_combination(y, V, E, A, "bogus")
