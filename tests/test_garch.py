import numpy as np
import pytest

from volgate.models.garch import GarchT


@pytest.mark.parametrize("name", ["garch_t", "egarch_t", "gjr_t"])
def test_fit_filter_and_causality(panel, name):
    m = GarchT(name)
    fit = m.fit(panel.iloc[:400])
    assert fit.converged and fit.nu > 2
    base = m.filter(fit, panel)
    assert base.shape == (len(panel),) and np.all(base > 0)
    pert = panel.copy()
    pert.iloc[450, pert.columns.get_loc("log_return")] *= 10
    out = m.filter(fit, pert)
    np.testing.assert_array_equal(base[:451], out[:451])
    assert base[451] != out[451]


def test_garch_recovers_simulated_persistence(panel):
    fit = GarchT("garch_t").fit(panel)
    p = fit.params
    assert 0.8 < p["alpha[1]"] + p["beta[1]"] < 1.0


def test_unknown_name_raises():
    with pytest.raises(KeyError):
        GarchT("figarch")
