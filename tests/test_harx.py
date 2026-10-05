import numpy as np
import pandas as pd
import pytest
from scipy import stats

from hart.harx import fit_harx, harx_loss_grad, harx_predict, harx_state, walk_forward_harx

A = 0.025


def _sim(n=20000, seed=0):
    rng = np.random.default_rng(seed)
    Z = rng.normal(size=(n, 10))
    b0 = np.zeros(10)
    b0[0], b0[3], b0[9] = 0.6, 0.3, 0.4
    log_s2 = -8.0 + Z @ b0
    y = np.exp(0.5 * log_s2) * rng.standard_normal(n)
    q = stats.norm.ppf(A)
    true = np.concatenate([[np.log(-q) - 4.0], 0.5 * b0, [np.log(stats.norm.pdf(q) / A / -q - 1)]])
    return Z, y, true


def test_gradient_matches_finite_differences():
    Z, y, true = _sim(2000)
    theta = true + np.random.default_rng(1).normal(0, 0.05, true.size)
    _, g = harx_loss_grad(theta, Z, y, A)
    h = 1e-6
    for i in range(theta.size):
        tp, tm = theta.copy(), theta.copy()
        tp[i] += h
        tm[i] -= h
        num = (harx_loss_grad(tp, Z, y, A)[0] - harx_loss_grad(tm, Z, y, A)[0]) / (2 * h)
        assert g[i] == pytest.approx(num, rel=1e-4, abs=1e-6)


def test_fit_recovers_parameters_and_orders_pair():
    Z, y, true = _sim()
    theta, ok = fit_harx(Z, y, A)
    assert ok
    np.testing.assert_allclose(theta[1:11], true[1:11], atol=0.06)
    v, e = harx_predict(theta, Z)
    assert np.all(e < v) and np.all(v < 0)


def test_fit_works_with_few_hits():
    Z, y, _ = _sim(1200, seed=3)
    theta, ok = fit_harx(Z, y, 0.01)
    v, e = harx_predict(theta, Z)
    assert np.all(np.isfinite(theta)) and np.all(e < v) and np.all(v < 0)


def test_state_columns_and_lookahead(panel):
    s = harx_state(panel)
    assert list(s.columns) == [f"{e}_{h}" for e in ("gk", "pk", "rs") for h in ("d", "w", "m")] + ["vix2"]
    j = 300
    pert = panel.copy()
    pert.iloc[j, pert.columns.get_indexer(["open", "high", "low", "close", "vix"])] *= 1.5
    out = harx_state(pert)
    pd.testing.assert_frame_equal(s.iloc[: j + 1], out.iloc[: j + 1])
    assert not s.iloc[j + 1].equals(out.iloc[j + 1])


def test_walk_forward_no_lookahead_and_nan_vix(panel):
    oos = panel.index[400].strftime("%Y-%m-%d")
    base = walk_forward_harx(panel, oos, 50, [A])
    assert (base["es_a025"] < base["var_a025"]).all()
    j = 470
    pert = panel.copy()
    pert.iloc[j, pert.columns.get_indexer(["log_return", "open", "high", "low", "close", "vix"])] *= 1.5
    out = walk_forward_harx(pert, oos, 50, [A])
    pd.testing.assert_frame_equal(base.loc[: panel.index[j]], out.loc[: panel.index[j]])
    p2 = panel.copy()
    p2.iloc[500, p2.columns.get_loc("vix")] = np.nan
    r = walk_forward_harx(p2, oos, 50, [A])
    assert np.isnan(r.loc[panel.index[501], "var_a025"])


def test_walk_forward_fallback_and_first_failure(panel, monkeypatch):
    import hart.harx as hx
    calls = {"n": 0}
    real = hx.fit_harx

    def flaky(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 2:
            return np.full(12, np.nan), False
        return real(*args, **kwargs)

    monkeypatch.setattr(hx, "fit_harx", flaky)
    out = walk_forward_harx(panel, panel.index[400].strftime("%Y-%m-%d"), 50, [A])
    flags = out.groupby("refit_date")["fallback_a025"].first().tolist()
    assert flags[:3] == [False, True, False]
    calls["n"] = 1  # next call is the 2nd -> fails on the first block
    with pytest.raises(RuntimeError, match="first fit"):
        walk_forward_harx(panel, panel.index[400].strftime("%Y-%m-%d"), 50, [A])
