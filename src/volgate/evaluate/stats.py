import numpy as np
import pandas as pd
from arch.bootstrap import MCS
from scipy import stats


def _bartlett_lags(n: int) -> int:
    return int(np.floor(4 * (n / 100.0) ** (2.0 / 9.0)))


def nw_variance(d, lags=None) -> float:
    d = np.asarray(d, dtype=float)
    n = len(d)
    lags = _bartlett_lags(n) if lags is None else lags
    u = d - d.mean()
    v = u @ u / n
    for k in range(1, lags + 1):
        v += 2.0 * (1.0 - k / (lags + 1.0)) * (u[k:] @ u[:-k]) / n
    return float(v)


def dm_test(l1, l2, h: int = 1):
    """Diebold-Mariano with HLN correction; negative stat means l1 has lower mean loss."""
    d = np.asarray(l1, dtype=float) - np.asarray(l2, dtype=float)
    n = len(d)
    v = nw_variance(d)
    if v <= 0:
        return np.nan, np.nan
    stat = d.mean() / np.sqrt(v / n) * np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    return float(stat), float(2 * stats.t.sf(abs(stat), df=n - 1))


def _xlogy(x, y):
    return x * np.log(y) if x > 0 else 0.0


def kupiec(hits, alpha):
    hits = np.asarray(hits, dtype=bool)
    n, x = len(hits), int(hits.sum())
    ll0 = _xlogy(n - x, 1 - alpha) + _xlogy(x, alpha)
    ll1 = _xlogy(n - x, 1 - x / n) + _xlogy(x, x / n)
    lr = max(-2.0 * (ll0 - ll1), 0.0)
    return float(lr), float(stats.chi2.sf(lr, 1))


def christoffersen(hits, alpha) -> dict:
    h = np.asarray(hits, dtype=int)
    prev, cur = h[:-1], h[1:]
    n00 = int(np.sum((prev == 0) & (cur == 0)))
    n01 = int(np.sum((prev == 0) & (cur == 1)))
    n10 = int(np.sum((prev == 1) & (cur == 0)))
    n11 = int(np.sum((prev == 1) & (cur == 1)))
    pi01 = n01 / (n00 + n01) if n00 + n01 else 0.0
    pi11 = n11 / (n10 + n11) if n10 + n11 else 0.0
    pi = (n01 + n11) / max(n00 + n01 + n10 + n11, 1)
    ll1 = _xlogy(n00, 1 - pi01) + _xlogy(n01, pi01) + _xlogy(n10, 1 - pi11) + _xlogy(n11, pi11)
    ll0 = _xlogy(n00 + n10, 1 - pi) + _xlogy(n01 + n11, pi)
    lr_ind = max(-2.0 * (ll0 - ll1), 0.0)
    lr_cc = lr_ind + kupiec(hits, alpha)[0]
    return {"lr_ind": float(lr_ind), "p_ind": float(stats.chi2.sf(lr_ind, 1)),
            "lr_cc": float(lr_cc), "p_cc": float(stats.chi2.sf(lr_cc, 2))}


def dq_test(hits, var, alpha, lags: int = 4):
    """Engle-Manganelli dynamic quantile test: regress Hit-alpha on lags and VaR."""
    h = np.asarray(hits, dtype=float) - alpha
    var = np.asarray(var, dtype=float)
    n = len(h)
    X = np.column_stack([np.ones(n - lags)] + [h[lags - k:n - k] for k in range(1, lags + 1)]
                        + [var[lags:]])
    y = h[lags:]
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    dq = float(beta @ X.T @ X @ beta / (alpha * (1 - alpha)))
    return dq, float(stats.chi2.sf(dq, X.shape[1]))


def mcneil_frey(y, var, es, reps: int = 2000, seed: int = 0):
    """Mean of (y - ES)/|ES| on VaR exceedances; one-sided bootstrap p for mean < 0."""
    y, var, es = (np.asarray(a, dtype=float) for a in (y, var, es))
    hit = y <= var
    r = (y[hit] - es[hit]) / np.abs(es[hit])
    if r.size < 2:
        return np.nan, np.nan, int(r.size)
    rng = np.random.default_rng(seed)
    c = r - r.mean()
    boot = c[rng.integers(0, r.size, (reps, r.size))].mean(axis=1)
    return float(r.mean()), float(np.mean(boot <= r.mean())), int(r.size)


def mcs_table(losses: pd.DataFrame, size: float = 0.10, reps: int = 2000, seed: int = 0) -> pd.DataFrame:
    """Model Confidence Set; methods with identical losses share the result of their first copy."""
    first = {}
    for c in losses.columns:
        match = next((k for k in first.values() if losses[c].equals(losses[k])), None)
        first[c] = match if match is not None else c
    unique = list(dict.fromkeys(first.values()))
    m = MCS(losses[unique], size=size, reps=reps, method="R", bootstrap="stationary", seed=seed)
    m.compute()
    p = m.pvalues["Pvalue"]
    return pd.DataFrame({"pvalue": [p[first[c]] for c in losses.columns],
                         "in_mcs": [first[c] in m.included for c in losses.columns]},
                        index=losses.columns)
