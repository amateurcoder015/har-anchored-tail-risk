import numpy as np

_DECAYED = ("W1", "Wo", "emb")


class CorrectionNet:
    """Bounded multiplicative correction of an anchor VaR/ES pair.

    VaR = VA*exp(cv); ES = VaR + (EA - VA)*exp(cs); c = bound*tanh(f(x)).
    kind="mlp": f = relu([x, emb(a)] W1 + b1) Wo + bo; kind="linear": f = x Wo + bo.
    """

    def __init__(self, n_features, n_assets, kind="mlp", hidden=16, emb=4, dropout=0.1,
                 alpha=0.025, lam_c=1.0, wd=1e-4, bound=0.25, lam_mean=0.0, seed=0):
        if kind not in ("mlp", "linear"):
            raise ValueError(f"bad kind {kind!r}")
        self.kind, self.F, self.alpha = kind, n_features, alpha
        self.dropout = dropout if kind == "mlp" else 0.0
        self.lam_c, self.wd, self.bound = lam_c, wd, bound
        self.lam_mean = lam_mean  # penalty on the batch-average correction (no level shift)
        rng = np.random.default_rng(seed)
        if kind == "mlp":
            d = n_features + emb
            self.p = {"W1": rng.normal(0, np.sqrt(2.0 / d), (d, hidden)), "b1": np.zeros(hidden),
                      "emb": rng.normal(0, 0.1, (n_assets, emb)),
                      "Wo": np.zeros((hidden, 2)), "bo": np.zeros(2)}
        else:
            self.p = {"Wo": np.zeros((n_features, 2)), "bo": np.zeros(2)}

    def forward(self, b, train=False, rng=None):
        p = self.p
        c = {}
        if self.kind == "mlp":
            z = np.concatenate([b["X"], p["emb"][b["a"]]], axis=1)
            a1 = z @ p["W1"] + p["b1"]
            h = np.maximum(a1, 0.0)
            if train and self.dropout > 0:
                mask = (rng.random(h.shape) >= self.dropout) / (1.0 - self.dropout)
            else:
                mask = np.ones_like(h)
            hd = h * mask
            c.update(z=z, a1=a1, mask=mask, inp=hd)
        else:
            c["inp"] = b["X"]
        th = np.tanh(c["inp"] @ p["Wo"] + p["bo"])
        cvs = self.bound * th
        c["th"], c["cv"], c["cs"] = th, cvs[:, 0], cvs[:, 1]
        c["D"] = b["EA"] - b["VA"]
        c["var"] = b["VA"] * np.exp(c["cv"])
        c["es"] = c["var"] + c["D"] * np.exp(c["cs"])
        return c

    def _fz(self, c, y):
        v, e, n = c["var"], c["es"], len(y)
        hit = (y <= v).astype(float)
        loss = np.mean(-hit * (v - y) / (self.alpha * e) + v / e + np.log(-e) - 1.0)
        dv = (-hit / (self.alpha * e) + 1.0 / e) / n
        de = (hit * (v - y) / (self.alpha * e**2) - v / e**2 + 1.0 / e) / n
        return loss, dv, de

    def data_loss(self, b):
        return float(self._fz(self.forward(b), b["y"])[0])

    def loss_and_grad(self, b, train=False, rng=None):
        p = self.p
        c = self.forward(b, train, rng)
        n = len(b["y"])
        loss, dv, de = self._fz(c, b["y"])
        dcv = (dv + de) * c["var"] + 2.0 * self.lam_c * c["cv"] / n
        dcs = de * c["D"] * np.exp(c["cs"]) + 2.0 * self.lam_c * c["cs"] / n
        dcv = dcv + 2.0 * self.lam_mean * c["cv"].mean() / n
        dcs = dcs + 2.0 * self.lam_mean * c["cs"].mean() / n
        dpre = np.column_stack([dcv, dcs]) * self.bound * (1.0 - c["th"] ** 2)
        g = {k: np.zeros_like(v) for k, v in p.items()}
        g["Wo"], g["bo"] = c["inp"].T @ dpre, dpre.sum(axis=0)
        if self.kind == "mlp":
            da1 = (dpre @ p["Wo"].T) * c["mask"] * (c["a1"] > 0)
            g["W1"], g["b1"] = c["z"].T @ da1, da1.sum(axis=0)
            np.add.at(g["emb"], b["a"], (da1 @ p["W1"].T)[:, self.F:])
        reg = self.lam_c * np.mean(c["cv"] ** 2 + c["cs"] ** 2)
        reg += self.lam_mean * (c["cv"].mean() ** 2 + c["cs"].mean() ** 2)
        for k in _DECAYED:
            if k in p:
                reg += 0.5 * self.wd * np.sum(p[k] ** 2)
                g[k] = g[k] + self.wd * p[k]
        return float(loss + reg), g
