import numpy as np
import pytest

from volgate.models.base import FitResult, tail_var_es
from volgate.models.simple import EWMA, HAR, VIXModel


@pytest.mark.parametrize("model", [EWMA(), HAR(), VIXModel()])
def test_fit_and_filter_shapes(panel, model):
    fit = model.fit(panel.iloc[:400])
    s2 = model.filter(fit, panel)
    assert s2.shape == (len(panel),)
    assert np.all(s2[100:] > 0)
    v, e = tail_var_es(fit, np.sqrt(s2[400:]), 0.025)
    assert np.all(e <= v) and np.all(v < 0)


@pytest.mark.parametrize("model", [EWMA(), HAR(), VIXModel()])
def test_filter_row_t_ignores_row_t(panel, model):
    fit = model.fit(panel.iloc[:400])
    base = model.filter(fit, panel)
    pert = panel.copy()
    pert.iloc[450, pert.columns.get_indexer(["log_return", "gk_var", "vix"])] *= 10
    out = model.filter(fit, pert)
    np.testing.assert_array_equal(base[:451], out[:451])
    assert base[451] != out[451]


def test_ewma_recursion_value(panel):
    m = EWMA(lam=0.94)
    s2 = m.filter(m.fit(panel.iloc[:400]), panel)
    r = panel["log_return"].to_numpy()
    assert s2[10] == pytest.approx(0.94 * s2[9] + 0.06 * r[9] ** 2)


def test_har_finite_with_zero_range_day(panel):
    p = panel.copy()
    p.iloc[300, p.columns.get_loc("gk_var")] = 1e-10  # floored zero-range day
    m = HAR()
    s2 = m.filter(m.fit(p.iloc[:400]), p)
    assert np.all(np.isfinite(s2[30:]))


def test_vix_model_nan_vix_gives_nan_not_error(panel):
    p = panel.copy()
    p.iloc[500, p.columns.get_loc("vix")] = np.nan
    m = VIXModel()
    s2 = m.filter(m.fit(p.iloc[:400]), p)
    assert np.isnan(s2[501]) and np.isfinite(s2[502])


def test_tail_uses_t_when_nu_set():
    fit = FitResult(params={}, z=None, nu=1e7, converged=True)
    v, _ = tail_var_es(fit, np.array([1.0]), 0.025)
    assert v[0] == pytest.approx(-1.959964, abs=1e-4)
