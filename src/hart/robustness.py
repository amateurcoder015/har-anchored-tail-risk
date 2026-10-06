import numpy as np
import pandas as pd

from volgate.evaluate.stats import nw_variance

Z_TARGET = 1.645 + 0.842  # one-sided 5% test with 80% power


def acerbi_szekely_z2(y, var, es, alpha) -> float:
    """Acerbi-Szekely (2014) Z2 for returns with VaR, ES < 0; about 0 if ES is right, negative if too shallow."""
    y, var, es = (np.asarray(a, dtype=float) for a in (y, var, es))
    hit = (y <= var).astype(float)
    return float(1.0 - np.sum(y * hit / es) / (len(y) * alpha))


def power_projection(D: pd.DataFrame, n_max: int = 1000) -> dict:
    """Project the pooled DM statistic for N assets from a dates x assets loss-differential panel.

    var(cross-asset mean) is approximated by s2 * (rho + (1 - rho) / N), with s2 the average
    per-asset long-run (Newey-West) variance and rho the average pairwise correlation.
    """
    D = D.dropna()
    T, N = D.shape
    s2 = float(np.mean([nw_variance(D[c].to_numpy()) for c in D.columns]))
    C = np.corrcoef(D.to_numpy().T)
    rho = float((C.sum() - N) / (N * (N - 1)))
    mean = float(D.to_numpy().mean())

    def t_of(n):
        return mean / np.sqrt(s2 * (rho + (1 - rho) / n) / T)

    need = next((n for n in range(1, n_max + 1) if abs(t_of(n)) >= Z_TARGET), None)
    return {"T": T, "N": N, "mean_diff": mean, "s2": s2, "rho": rho, "t_at_N": float(t_of(N)),
            "t_limit": float(mean / np.sqrt(s2 * max(rho, 0.0) / T)) if rho > 0 else float("inf"),
            "n_required": need}


def sign_counts(per_asset: dict, lower: str, higher: str) -> tuple[int, int]:
    """(assets where `lower` has lower mean loss than `higher`, total assets)."""
    wins = sum(int(L[lower].mean() < L[higher].mean()) for L in per_asset.values())
    return wins, len(per_asset)
