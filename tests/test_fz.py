import numpy as np
import pytest
from scipy import stats

from volgate.risk.fz import fz0_loss


def test_known_value_with_hit():
    # hit: y <= var. Loss = -(v - y)/(a*e) + v/e + log(-e) - 1
    loss = fz0_loss(-0.05, -0.03, -0.04, 0.025)
    assert loss == pytest.approx(20 + 0.75 + np.log(0.04) - 1, rel=1e-12)


def test_known_value_without_hit():
    loss = fz0_loss(0.01, -0.03, -0.04, 0.025)
    assert loss == pytest.approx(0.75 + np.log(0.04) - 1, rel=1e-12)


def test_nonnegative_es_raises():
    with pytest.raises(ValueError, match="ES must be negative"):
        fz0_loss(np.array([0.0]), np.array([-0.01]), np.array([0.0]), 0.025)


def test_true_quantile_and_es_minimise_expected_loss():
    rng = np.random.default_rng(0)
    y = rng.standard_normal(400_000)
    a = 0.025
    v = stats.norm.ppf(a)
    e = -stats.norm.pdf(v) / a
    true = fz0_loss(y, v, e, a).mean()
    for dv, de in [(1.1, 1.0), (0.9, 1.0), (1.0, 1.1), (1.0, 0.9)]:
        assert fz0_loss(y, v * dv, e * de, a).mean() > true
