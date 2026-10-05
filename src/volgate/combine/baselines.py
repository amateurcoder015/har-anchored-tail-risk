import numpy as np
from scipy.optimize import minimize

from volgate.risk.fz import fz0_loss


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def combine_pairs(V, E, wq, ws) -> tuple[np.ndarray, np.ndarray]:
    """Taylor (2020) form: VaR = sum wq*VaR_i; ES = VaR + sum ws*(ES_i - VaR_i)."""
    var = np.sum(V * wq, axis=-1)
    return var, var + np.sum((E - V) * ws, axis=-1)


def mean_combo(V, E):
    M = V.shape[1]
    w = np.full(M, 1.0 / M)
    return combine_pairs(V, E, w, w)


def median_combo(V, E):
    return np.median(V, axis=1), np.median(E, axis=1)


def _score(y, V, E, alpha):
    return np.array([fz0_loss(y, V[:, i], E[:, i], alpha).mean() for i in range(V.shape[1])])


def fit_min_score(y, V, E, alpha, init=None, n_random=3, seed=0):
    """Taylor (2020) minimum score combining, weights by softmax parameters."""
    M = V.shape[1]

    def obj(theta):
        v, e = combine_pairs(V, E, _softmax(theta[:M]), _softmax(theta[M:]))
        return fz0_loss(y, v, e, alpha).mean()

    rng = np.random.default_rng(seed)
    starts = [np.zeros(2 * M)] + ([init] if init is not None else [])
    starts += [rng.normal(0, 1, 2 * M) for _ in range(n_random)]
    best = min((minimize(obj, s, method="Powell") for s in starts), key=lambda r: r.fun)
    return _softmax(best.x[:M]), _softmax(best.x[M:]), best.x


def fit_relative_score(y, V, E, alpha):
    """Taylor (2020) relative score combining: w_i proportional to exp(-lam * S_i)."""
    S = _score(y, V, E, alpha)
    grid = np.concatenate([[0.0], np.logspace(-2, 4, 400)])
    best = (np.inf, 0.0, None)
    for lam in grid:
        w = _softmax(-lam * S)
        v, e = combine_pairs(V, E, w, w)
        s = fz0_loss(y, v, e, alpha).mean()
        if s < best[0]:
            best = (s, lam, w)
    return best[2], best[1]


def fit_previous_best(y, V, E, alpha):
    w = np.zeros(V.shape[1])
    w[np.argmin(_score(y, V, E, alpha))] = 1.0
    return w


def rolling_combination(y, V, E, alpha, method, refit_every=21, min_train=250):
    """Weights refitted every `refit_every` rows on all complete rows before the block."""
    if method not in ("min_score", "relative_score", "previous_best"):
        raise ValueError(f"unknown method {method!r}")
    n, M = V.shape
    var, es = np.full(n, np.nan), np.full(n, np.nan)
    WQ = np.full((n, M), np.nan)
    ok = np.isfinite(y) & np.isfinite(V).all(axis=1) & np.isfinite(E).all(axis=1)
    theta = None
    for k in range(min_train, n, refit_every):
        tr = ok[:k]
        yt, Vt, Et = y[:k][tr], V[:k][tr], E[:k][tr]
        if method == "min_score":
            wq, ws, theta = fit_min_score(yt, Vt, Et, alpha, init=theta,
                                          n_random=3 if theta is None else 0)
        elif method == "relative_score":
            wq, _ = fit_relative_score(yt, Vt, Et, alpha)
            ws = wq
        else:
            wq = ws = fit_previous_best(yt, Vt, Et, alpha)
        end = min(k + refit_every, n)
        var[k:end], es[k:end] = combine_pairs(V[k:end], E[k:end], wq, ws)
        WQ[k:end] = wq
    return var, es, WQ
