import numpy as np
import pytest

from volgate.risk.dist import alpha_tag, empirical_var_es, student_t_var_es


def test_alpha_tag():
    assert alpha_tag(0.025) == "a025"
    assert alpha_tag(0.01) == "a010"
    assert alpha_tag(0.05) == "a050"


def test_student_t_large_nu_matches_normal():
    v, e = student_t_var_es(np.array([2.0]), 1e7, 0.025)
    assert v[0] == pytest.approx(2 * -1.959964, abs=1e-4)
    assert e[0] == pytest.approx(2 * -2.337803, abs=1e-4)


def test_student_t_matches_monte_carlo():
    nu, a = 5.0, 0.025
    rng = np.random.default_rng(1)
    x = rng.standard_t(nu, 2_000_000) * np.sqrt((nu - 2) / nu)
    q = np.quantile(x, a)
    v, e = student_t_var_es(1.0, nu, a)
    assert float(v) == pytest.approx(q, rel=0.01)
    assert float(e) == pytest.approx(x[x <= q].mean(), rel=0.01)


def test_student_t_rejects_small_nu():
    with pytest.raises(ValueError, match="nu"):
        student_t_var_es(1.0, 2.0, 0.025)


def test_empirical_var_es_known_values():
    z = np.linspace(-1, 1, 1001)
    v, e = empirical_var_es(z, np.array([2.0]), 0.025)
    assert v[0] == pytest.approx(2 * -0.95)
    assert e[0] == pytest.approx(2 * -0.975)


def test_empirical_ignores_nan_and_requires_enough_points():
    z = np.concatenate([np.linspace(-1, 1, 1001), [np.nan]])
    empirical_var_es(z, 1.0, 0.025)
    with pytest.raises(ValueError, match="residuals"):
        empirical_var_es(np.linspace(-1, 1, 50), 1.0, 0.025)


def test_es_below_var():
    v, e = student_t_var_es(np.array([0.01, 0.02]), 6.0, 0.01)
    assert np.all(e <= v) and np.all(v < 0)
