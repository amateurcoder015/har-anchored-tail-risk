import numpy as np
import pytest
from scipy import stats

from hart.correction import CorrectionNet
from volgate.gate.net import train_gate

A = 0.025


def _batch(n=80, F=4, n_assets=2, seed=0):
    rng = np.random.default_rng(seed)
    s = np.exp(rng.normal(-4, 0.3, n))
    q = stats.norm.ppf(A)
    VA = s * q
    EA = -s * stats.norm.pdf(q) / A
    return {"X": rng.normal(size=(n, F)), "a": rng.integers(0, n_assets, n), "VA": VA, "EA": EA,
            "y": rng.normal(0, 0.03, n)}


@pytest.mark.parametrize("kind", ["mlp", "linear"])
@pytest.mark.parametrize("dropout", [0.0, 0.3])
def test_gradients_match_finite_differences(kind, dropout):
    b = _batch()
    net = CorrectionNet(4, 2, kind=kind, hidden=5, emb=2, dropout=dropout, lam_c=0.7, wd=1e-3, seed=1)
    rng = np.random.default_rng(2)
    for k in net.p:
        net.p[k] = net.p[k] + rng.normal(0, 0.3, net.p[k].shape)
    out = net.forward(b)
    hits = (b["y"] <= out["var"]).sum()
    assert 0 < hits < len(b["y"])

    def loss(grad=False):
        res = net.loss_and_grad(b, train=dropout > 0, rng=np.random.default_rng(7))
        return res if grad else res[0]

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


@pytest.mark.parametrize("kind", ["mlp", "linear"])
def test_untrained_equals_anchor(kind):
    b = _batch()
    out = CorrectionNet(4, 2, kind=kind).forward(b)
    np.testing.assert_allclose(out["var"], b["VA"])
    np.testing.assert_allclose(out["es"], b["EA"])


def test_bounds_and_ordering_with_extreme_params():
    b = _batch()
    b["EA"][0] = b["VA"][0]  # zero ES spacing on one row
    net = CorrectionNet(4, 2, seed=0)
    for k in net.p:
        net.p[k] = net.p[k] + 50.0
    out = net.forward(b)
    assert np.all(np.abs(out["cv"]) <= 0.25 + 1e-12) and np.all(np.abs(out["cs"]) <= 0.25 + 1e-12)
    assert np.all(out["es"] <= out["var"]) and np.all(out["var"] < 0)
    assert out["es"][0] == out["var"][0]


def test_linear_ignores_embedding():
    net = CorrectionNet(4, 2, kind="linear", emb=3)
    assert "emb" not in net.p and "W1" not in net.p


def test_learns_state_dependent_scale():
    rng = np.random.default_rng(0)
    n = 4000
    x = rng.integers(0, 2, n).astype(float)
    sd = np.where(x == 1, 0.012, 0.01)  # anchor assumes 0.01; true risk 20% higher when x = 1
    y = rng.normal(0, sd)
    q = stats.norm.ppf(A)
    b = {"X": x[:, None], "a": np.zeros(n, int), "VA": np.full(n, 0.01 * q),
         "EA": np.full(n, -0.01 * stats.norm.pdf(q) / A), "y": y}
    tr = {k: v[:3000] for k, v in b.items()}
    va = {k: v[3000:] for k, v in b.items()}
    net = CorrectionNet(1, 1, emb=0, dropout=0.0, lam_c=0.0, seed=0)
    base = net.data_loss(va)
    info = train_gate(net, tr, va, lr=1e-2, epochs=100, patience=20, seed=0)
    assert info["best_val"] < base
    cv = net.forward(va)["cv"]
    assert cv[va["X"][:, 0] == 1].mean() > cv[va["X"][:, 0] == 0].mean() + 0.1
