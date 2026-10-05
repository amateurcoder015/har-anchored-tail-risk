import numpy as np
import pytest
from scipy import stats

from volgate.gate.net import GateNet, train_gate

A = 0.025


def _batch(n=64, F=5, M=3, n_assets=2, seed=0):
    rng = np.random.default_rng(seed)
    s = np.exp(rng.normal(-4, 0.3, (n, M)))
    q = stats.norm.ppf(A)
    return {"X": rng.normal(size=(n, F)), "a": rng.integers(0, n_assets, n),
            "V": s * q, "E": -s * stats.norm.pdf(q) / A, "S2": s**2,
            "y": rng.normal(0, 0.02, n)}


CONFIGS = [dict(mode="pairs", use_scale=True), dict(mode="pairs", use_scale=True, lam_g=2.0),
           dict(mode="pairs", use_scale=False),
           dict(mode="variance", use_scale=False, nu=6.0),
           dict(mode="variance", loss="qlike", use_scale=False, nu=6.0)]


@pytest.mark.parametrize("dropout", [0.0, 0.3])
@pytest.mark.parametrize("cfg", CONFIGS)
def test_gradients_match_finite_differences(cfg, dropout):
    b = _batch()
    b["y"] = b["y"] * 3  # enough VaR hits to exercise the hit branch
    net = GateNet(5, 2, 3, hidden=4, emb=2, dropout=dropout, lam_eq=0.5, wd=1e-3, seed=1, **cfg)
    rng = np.random.default_rng(2)
    for k in net.p:  # move heads off zero so every path is exercised
        net.p[k] = net.p[k] + rng.normal(0, 0.3, net.p[k].shape)
    hits = (b["y"] <= net.forward(b)["var"]).sum()
    assert 0 < hits < len(b["y"])

    def loss(grad=False):  # same dropout mask on every call
        out = net.loss_and_grad(b, train=dropout > 0, rng=np.random.default_rng(7))
        return out if grad else out[0]

    _, g = loss(grad=True)
    h = 1e-6
    for k, v in net.p.items():
        for idx in np.ndindex(v.shape):
            old = v[idx]
            v[idx] = old + h
            lp = loss()
            v[idx] = old - h
            lm = loss()
            v[idx] = old
            assert g[k][idx] == pytest.approx((lp - lm) / (2 * h), rel=1e-4, abs=1e-6), (k, idx)


def test_untrained_gate_equals_equal_weights():
    b = _batch()
    net = GateNet(5, 2, 3, seed=0)
    out = net.forward(b)
    np.testing.assert_allclose(out["var"], b["V"].mean(axis=1))
    np.testing.assert_allclose(out["es"], b["E"].mean(axis=1))


def test_es_below_var_after_random_params():
    b = _batch()
    net = GateNet(5, 2, 3, seed=0)
    rng = np.random.default_rng(3)
    for k in net.p:
        net.p[k] = net.p[k] + rng.normal(0, 2, net.p[k].shape)
    out = net.forward(b)
    assert np.all(out["es"] <= out["var"]) and np.all(out["var"] < 0)


def test_gate_learns_state_dependent_weights():
    rng = np.random.default_rng(0)
    n = 4000
    x = rng.integers(0, 2, n).astype(float)
    sd = np.where(x == 1, 0.01, 0.03)
    y = rng.normal(0, sd)
    q, es = stats.norm.ppf(A), -stats.norm.pdf(stats.norm.ppf(A)) / A
    V = np.column_stack([np.full(n, 0.01 * q), np.full(n, 0.03 * q)])
    E = np.column_stack([np.full(n, 0.01 * es), np.full(n, 0.03 * es)])
    b = {"X": x[:, None], "a": np.zeros(n, int), "V": V, "E": E, "S2": (V / q) ** 2, "y": y}
    tr = {k: v[:3000] for k, v in b.items()}
    va = {k: v[3000:] for k, v in b.items()}
    net = GateNet(1, 1, 2, hidden=8, emb=0, dropout=0.0, seed=0)
    eq = net.data_loss(va)
    info = train_gate(net, tr, va, lr=1e-2, epochs=100, patience=20, seed=0)
    assert info["best_val"] < eq - 0.05
    w = net.forward(va)["wq"]
    assert w[va["X"][:, 0] == 1, 0].mean() > 0.8


def test_train_restores_best_epoch(monkeypatch):
    b = _batch(200)
    net = GateNet(5, 2, 3, seed=0)
    start = {k: v.copy() for k, v in net.p.items()}
    calls = {"n": 0}

    def worse_each_time(self, batch):
        calls["n"] += 1
        return float(calls["n"])  # epoch 0 is best

    monkeypatch.setattr(GateNet, "data_loss", worse_each_time)
    info = train_gate(net, b, b, epochs=10, patience=3, seed=0)
    assert info["best_epoch"] == 0
    assert not all(np.array_equal(start[k], net.p[k]) for k in net.p)  # trained one epoch first


def test_training_is_deterministic():
    b = _batch(300)
    outs = []
    for _ in range(2):
        net = GateNet(5, 2, 3, seed=4)
        train_gate(net, b, b, epochs=5, seed=4)
        outs.append(net.forward(b)["var"])
    np.testing.assert_array_equal(outs[0], outs[1])


def test_invalid_config_raises():
    with pytest.raises(ValueError):
        GateNet(5, 2, 3, mode="pairs", loss="qlike")
    with pytest.raises(ValueError):
        GateNet(5, 2, 3, mode="variance", nu=None)


def test_scale_shrinkage_pulls_g_toward_one():
    b = _batch(400)
    b["y"] = b["y"] * 0.3  # forecasts far too wide: unpenalised gate shrinks g
    gs = []
    for lam_g in (0.0, 50.0):
        net = GateNet(5, 2, 3, dropout=0.0, lam_g=lam_g, seed=0)
        train_gate(net, b, b, lr=1e-2, epochs=60, patience=60, seed=0)
        gs.append(abs(np.log(net.forward(b)["g"])).mean())
    assert gs[1] < 0.5 * gs[0]
