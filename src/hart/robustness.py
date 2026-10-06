import numpy as np
import pandas as pd

from volgate.evaluate.stats import nw_variance

Z_TARGET = 1.645 + 0.842  # one-sided 5% test with 80% power


def acerbi_szekely_z2(y, var, es, alpha) -> float:
    """Acerbi-Szekely (2014) Z2 for returns with VaR, ES < 0; about 0 if ES is right, negative if too shallow."""
    y, var, es = (np.asarray(a, dtype=float) for a in (y, var, es))
    hit = (y <= var).astype(float)
    return float(1.0 - np.sum(y * hit / es) / (len(y) * alpha))


def z2_critical(T: int, alpha: float, reps: int = 5000, seed: int = 0, level: float = 0.05) -> float:
    """Simulated lower `level` quantile of Z2 under correct normal VaR/ES with T observations."""
    from scipy import stats
    rng = np.random.default_rng(seed)
    q = stats.norm.ppf(alpha)
    es = -stats.norm.pdf(q) / alpha
    y = rng.standard_normal((reps, T))
    z2 = 1.0 - (y * (y <= q)).sum(axis=1) / es / (T * alpha)
    return float(np.quantile(z2, level))


def power_projection(D: pd.DataFrame, n_max: int = 1000) -> dict:
    """Project the pooled DM statistic for N assets from a dates x assets loss-differential panel.

    var(cross-asset mean) is modelled as s2 * (rho + (1 - rho) / N) / T, with s2 the average
    per-asset long-run (Newey-West) variance; rho is calibrated so the formula matches the
    Newey-West variance of the observed pooled series at the observed N. Approximation:
    assumes equicorrelated assets. n_required is the smallest N with t <= -Z_TARGET (first
    method lower), ignoring multiple-testing adjustment; None if no N up to n_max suffices.
    """
    D = D.dropna()
    T, N = D.shape
    s2 = float(np.mean([nw_variance(D[c].to_numpy()) for c in D.columns]))
    v_pool = nw_variance(D.mean(axis=1).to_numpy())
    rho = float(np.clip((v_pool / s2 - 1.0 / N) / (1.0 - 1.0 / N), 0.0, 1.0))
    mean = float(D.to_numpy().mean())

    def t_of(n):
        return mean / np.sqrt(s2 * (rho + (1 - rho) / n) / T)

    need = next((n for n in range(1, n_max + 1) if t_of(n) <= -Z_TARGET), None)
    return {"T": T, "N": N, "mean_diff": mean, "s2": s2, "rho": rho, "t_at_N": float(t_of(N)),
            "t_limit": float(mean / np.sqrt(s2 * max(rho, 0.0) / T)) if rho > 0 else float("inf"),
            "n_required": need}


def sign_counts(per_asset: dict, lower: str, higher: str) -> tuple[int, int]:
    """(assets where `lower` has lower mean loss than `higher`, total assets)."""
    wins = sum(int(L[lower].mean() < L[higher].mean()) for L in per_asset.values())
    return wins, len(per_asset)
