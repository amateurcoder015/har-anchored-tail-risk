import numpy as np

from volgate.risk.dist import student_t_var_es

_DECAYED = ("W1", "Wq", "Ws", "wb", "emb")


def _softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def _softmax_back(w, dw):
    return w * (dw - np.sum(dw * w, axis=1, keepdims=True))


class GateNet:
    """One-hidden-layer gate over M base models with hand-written gradients.

    pairs mode: VaR = g*sum(wq*V), ES = g*(sum(wq*V) + sum(ws*(E-V))), g = exp(0.5*tanh(.))
    variance mode: s2 = sum(wq*S2); VaR/ES = Student-t(nu) quantile/ES * sqrt(s2)
    """

    def __init__(self, n_features, n_assets, n_models, hidden=16, emb=4, dropout=0.1,
                 mode="pairs", loss="fz0", use_scale=True, alpha=0.025, nu=None,
                 lam_eq=0.0, wd=1e-4, lam_g=0.0, seed=0):
        if mode not in ("pairs", "variance"):
            raise ValueError(f"bad mode {mode!r}")
        if loss not in ("fz0", "qlike"):
            raise ValueError(f"bad loss {loss!r}")
        if loss == "qlike" and mode != "variance":
            raise ValueError("qlike loss requires variance mode")
        if mode == "variance" and nu is None:
            raise ValueError("variance mode needs nu")
        self.F, self.M = n_features, n_models
        self.mode, self.loss, self.alpha = mode, loss, alpha
        self.use_scale = use_scale and mode == "pairs"
        self.dropout, self.lam_eq, self.wd = dropout, lam_eq, wd
        self.lam_g = lam_g  # penalty on (log g)^2, pulls the scale factor toward 1
        rng = np.random.default_rng(seed)
        d = n_features + emb
        self.p = {
            "W1": rng.normal(0, np.sqrt(2.0 / d), (d, hidden)), "b1": np.zeros(hidden),
            "emb": rng.normal(0, 0.1, (n_assets, emb)),
            "Wq": np.zeros((hidden, n_models)), "bq": np.zeros(n_models),
        }
        if mode == "pairs":
            self.p["Ws"] = np.zeros((hidden, n_models))
            self.p["bs"] = np.zeros(n_models)
            if self.use_scale:
                self.p["wb"] = np.zeros(hidden)
                self.p["bb"] = np.zeros(1)
        else:
            qv, qe = student_t_var_es(1.0, nu, alpha)
            self.qv, self.qe = float(qv), float(qe)

    def forward(self, b, train=False, rng=None):
        p = self.p
        z = np.concatenate([b["X"], p["emb"][b["a"]]], axis=1)
        a1 = z @ p["W1"] + p["b1"]
        h = np.maximum(a1, 0.0)
        if train and self.dropout > 0:
            mask = (rng.random(h.shape) >= self.dropout) / (1.0 - self.dropout)
        else:
            mask = np.ones_like(h)
        hd = h * mask
        c = {"z": z, "a1": a1, "mask": mask, "hd": hd,
             "wq": _softmax(hd @ p["Wq"] + p["bq"])}
        if self.mode == "pairs":
            c["ws"] = _softmax(hd @ p["Ws"] + p["bs"])
            if self.use_scale:
                c["th"] = np.tanh(hd @ p["wb"] + p["bb"][0])
                c["g"] = np.exp(0.5 * c["th"])
            else:
                c["g"] = np.ones(len(hd))
            c["v0"] = np.sum(b["V"] * c["wq"], axis=1)
            c["sp"] = np.sum((b["E"] - b["V"]) * c["ws"], axis=1)
            c["var"] = c["g"] * c["v0"]
            c["es"] = c["g"] * (c["v0"] + c["sp"])
        else:
            c["s2"] = np.sum(b["S2"] * c["wq"], axis=1)
            c["sd"] = np.sqrt(c["s2"])
            c["var"], c["es"] = self.qv * c["sd"], self.qe * c["sd"]
        return c

    def _data_terms(self, c, y):
        n = len(y)
        if self.loss == "fz0":
            v, e = c["var"], c["es"]
            hit = (y <= v).astype(float)
            loss = np.mean(-hit * (v - y) / (self.alpha * e) + v / e + np.log(-e) - 1.0)
            dv = (-hit / (self.alpha * e) + 1.0 / e) / n
            de = (hit * (v - y) / (self.alpha * e**2) - v / e**2 + 1.0 / e) / n
            return loss, dv, de, None
        y2 = np.maximum(y**2, 1e-12)
        ratio = y2 / c["s2"]
        return np.mean(ratio - np.log(ratio) - 1.0), None, None, (-y2 / c["s2"]**2 + 1.0 / c["s2"]) / n

    def data_loss(self, b):
        return float(self._data_terms(self.forward(b), b["y"])[0])

    def loss_and_grad(self, b, train=False, rng=None):
        p, M = self.p, self.M
        c = self.forward(b, train, rng)
        n = len(b["y"])
        loss, dv, de, ds2 = self._data_terms(c, b["y"])
        g = {k: np.zeros_like(v) for k, v in p.items()}
        eq = 1.0 / M
        pen = np.mean(np.sum((c["wq"] - eq) ** 2, axis=1))
        dwq = 2.0 * self.lam_eq * (c["wq"] - eq) / n
        dhd = np.zeros_like(c["hd"])
        if self.mode == "pairs":
            pen += np.mean(np.sum((c["ws"] - eq) ** 2, axis=1))
            gg = c["g"]
            dwq = dwq + ((dv + de) * gg)[:, None] * b["V"]
            dws = 2.0 * self.lam_eq * (c["ws"] - eq) / n + (de * gg)[:, None] * (b["E"] - b["V"])
            dls = _softmax_back(c["ws"], dws)
            g["Ws"], g["bs"] = c["hd"].T @ dls, dls.sum(axis=0)
            dhd = dhd + dls @ p["Ws"].T
            if self.use_scale:
                dgg = dv * c["v0"] + de * (c["v0"] + c["sp"])
                dpre = (dgg * gg * 0.5 + self.lam_g * 0.5 * c["th"] / n) * (1.0 - c["th"] ** 2)
                g["wb"], g["bb"] = c["hd"].T @ dpre, np.array([dpre.sum()])
                dhd = dhd + dpre[:, None] * p["wb"][None, :]
        else:
            if self.loss == "fz0":
                ds2 = (dv * self.qv + de * self.qe) / (2.0 * c["sd"])
            dwq = dwq + ds2[:, None] * b["S2"]
        dlq = _softmax_back(c["wq"], dwq)
        g["Wq"], g["bq"] = c["hd"].T @ dlq, dlq.sum(axis=0)
        dhd = dhd + dlq @ p["Wq"].T
        da1 = dhd * c["mask"] * (c["a1"] > 0)
        g["W1"], g["b1"] = c["z"].T @ da1, da1.sum(axis=0)
        np.add.at(g["emb"], b["a"], (da1 @ p["W1"].T)[:, self.F:])
        reg = 0.0
        for k in _DECAYED:
            if k in p:
                reg += 0.5 * self.wd * np.sum(p[k] ** 2)
                g[k] = g[k] + self.wd * p[k]
        if self.use_scale:
            reg += self.lam_g * np.mean((0.5 * c["th"]) ** 2)
        return float(loss + self.lam_eq * pen + reg), g


def _take(b, idx):
    return {k: v[idx] for k, v in b.items()}


def train_gate(net, tr, va, lr=1e-3, epochs=300, batch=256, patience=20, seed=0):
    """Adam with early stopping on validation data loss; restores the best parameters."""
    rng = np.random.default_rng(seed)
    m1 = {k: np.zeros_like(v) for k, v in net.p.items()}
    m2 = {k: np.zeros_like(v) for k, v in net.p.items()}
    t, n = 0, len(tr["y"])
    best_val, best_p, best_ep, hist = np.inf, None, -1, []
    for ep in range(epochs):
        order = rng.permutation(n)
        for i in range(0, n, batch):
            _, g = net.loss_and_grad(_take(tr, order[i:i + batch]), train=True, rng=rng)
            t += 1
            for k in net.p:
                m1[k] = 0.9 * m1[k] + 0.1 * g[k]
                m2[k] = 0.999 * m2[k] + 0.001 * g[k] ** 2
                net.p[k] = net.p[k] - lr * (m1[k] / (1 - 0.9**t)) / (np.sqrt(m2[k] / (1 - 0.999**t)) + 1e-8)
        val = net.data_loss(va)
        hist.append(val)
        if val < best_val:
            best_val, best_ep = val, ep
            best_p = {k: v.copy() for k, v in net.p.items()}
        elif ep - best_ep >= patience:
            break
    net.p = best_p
    return {"best_val": best_val, "best_epoch": best_ep, "history": hist}
