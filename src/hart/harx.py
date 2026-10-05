import numpy as np
import pandas as pd
from scipy.optimize import minimize

from hart.ranges import RANGE_FLOOR, parkinson, rogers_satchell
from volgate.data.panel import gk_variance
from volgate.risk.dist import alpha_tag

ESTIMATORS = {"gk": gk_variance, "pk": parkinson, "rs": rogers_satchell}
STATE_COLUMNS = [f"{e}_{h}" for e in ESTIMATORS for h in ("d", "w", "m")] + ["vix2"]


def harx_state(df: pd.DataFrame) -> pd.DataFrame:
    """Log range variances (daily, 5-day, 22-day means) and log VIX variance, through t-1."""
    s = pd.DataFrame(index=df.index)
    for name, fn in ESTIMATORS.items():
        v = fn(df).clip(lower=RANGE_FLOOR).shift(1)
        m = v.rolling(22).mean()
        # A zero-range day (e.g. Rogers-Satchell on a trend day) would give log(1e-10);
        # bound the daily term below at a tenth of its own 22-day mean.
        s[f"{name}_d"] = np.log(np.maximum(v, 0.1 * m))
        s[f"{name}_w"] = np.log(v.rolling(5).mean())
        s[f"{name}_m"] = np.log(m)
    s["vix2"] = np.log((df["vix"].shift(1) / 100.0) ** 2 / 252.0)
    return s


def harx_predict(theta, Z):
    u = theta[0] + Z @ theta[1:-1]
    var = -np.exp(u)
    return var, var * (1.0 + np.exp(theta[-1]))


def harx_loss_grad(theta, Z, y, alpha):
    """Mean FZ0 with VaR = -exp(u), ES = k*VaR, k = 1 + exp(c); analytic gradient."""
    u = theta[0] + Z @ theta[1:-1]
    v = -np.exp(u)
    k = 1.0 + np.exp(theta[-1])
    hit = (y <= v).astype(float)
    r = y / v
    L = -hit * (1.0 - r) / (alpha * k) + 1.0 / k + u + np.log(k) - 1.0
    dL_du = 1.0 - hit * r / (alpha * k)
    dL_dk = hit * (1.0 - r) / (alpha * k**2) - 1.0 / k**2 + 1.0 / k
    n = len(y)
    g = np.empty_like(theta)
    g[0] = dL_du.mean()
    g[1:-1] = Z.T @ dL_du / n
    g[-1] = dL_dk.mean() * (k - 1.0)
    return float(L.mean()), g


def _init(Z, y, alpha):
    X = np.column_stack([np.ones(len(y)), Z])
    beta, *_ = np.linalg.lstsq(X, np.log(y**2 + 1e-12), rcond=None)
    sd = np.exp(0.5 * (X @ beta))
    zq = y / sd
    q = np.quantile(zq, alpha)
    es = zq[zq <= q].mean()
    ratio = max(es / q - 1.0, 0.05)
    return np.concatenate([[np.log(-q) + 0.5 * beta[0]], 0.5 * beta[1:], [np.log(ratio)]])


def fit_harx(Z, y, alpha, init=None, n_starts=3, seed=0):
    rng = np.random.default_rng(seed)
    base = _init(Z, y, alpha) if init is None else np.asarray(init, dtype=float)
    starts = [base] + [base + rng.normal(0, 0.1, base.size) for _ in range(n_starts - 1)]
    best = None
    for s in starts:
        res = minimize(harx_loss_grad, s, args=(Z, y, alpha), jac=True, method="L-BFGS-B")
        if np.all(np.isfinite(res.x)) and (best is None or res.fun < best.fun):
            best = res
    if best is None:
        return np.full(base.size, np.nan), False
    return best.x, True


def walk_forward_harx(df, oos_start, refit_every, alphas):
    """Expanding-window refit every `refit_every` rows; Z standardized on the training rows."""
    df = df.sort_index()
    S = harx_state(df).to_numpy(dtype=float)
    y = df["log_return"].to_numpy(dtype=float)
    ok = np.isfinite(S).all(axis=1) & np.isfinite(y)
    first = int(np.flatnonzero(df.index >= pd.Timestamp(oos_start))[0])
    out = pd.DataFrame(index=df.index[first:])
    for a in alphas:
        t = alpha_tag(a)
        var = np.full(len(df), np.nan)
        es = np.full(len(df), np.nan)
        fb = np.zeros(len(df), dtype=bool)
        prev = None
        for k in range(first, len(df), refit_every):
            end = min(k + refit_every, len(df))
            tr = ok[:k]
            mu, sd = S[:k][tr].mean(axis=0), S[:k][tr].std(axis=0)
            sd = np.where(sd < 1e-12, 1.0, sd)
            theta, success = fit_harx((S[:k][tr] - mu) / sd, y[:k][tr], a,
                                      init=None if prev is None else prev[0])
            if not success:
                if prev is None:
                    raise RuntimeError("R1: first fit failed")
                theta, mu, sd = prev
                fb[k:end] = True
            prev = (theta, mu, sd)
            var[k:end], es[k:end] = harx_predict(theta, (S[k:end] - mu) / sd)
        out[f"var_{t}"], out[f"es_{t}"] = var[first:], es[first:]
        out[f"fallback_{t}"] = fb[first:]
    out["refit_date"] = [df.index[first + ((i // refit_every) * refit_every)] for i in range(len(out))]
    return out
