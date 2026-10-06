import numpy as np
import pandas as pd

from volgate.evaluate.stats import dm_test

# (name, method expected to have LOWER loss, method expected to have HIGHER loss)
HYPOTHESES = [
    ("H1", "r0", "mean"),
    ("H2", "r0", "min_score"),
    ("H3", "r0", "gate"),
    ("H4", "gate_v2", "gate"),
]


def one_sided_p(stat: float, p_two: float) -> float:
    """One-sided p-value for 'first method has lower loss' from a two-sided DM test."""
    if np.isnan(stat) or np.isnan(p_two):
        return np.nan
    return p_two / 2.0 if stat < 0 else 1.0 - p_two / 2.0


def holm(pvalues: dict, level: float = 0.05) -> pd.DataFrame:
    """Holm step-down adjustment; NaN p-values are never rejected."""
    names = list(pvalues)
    p = np.array([pvalues[k] for k in names], dtype=float)
    order = np.argsort(np.where(np.isnan(p), np.inf, p))
    m = len(p)
    adj = np.full(m, np.nan)
    running = 0.0
    for rank, i in enumerate(order):
        if np.isnan(p[i]):
            continue
        running = max(running, min(1.0, (m - rank) * p[i]))
        adj[i] = running
    out = pd.DataFrame({"p": p, "p_holm": adj}, index=names)
    out["reject"] = out["p_holm"] <= level
    return out


def evaluate_hypotheses(losses: pd.DataFrame, level: float = 0.05) -> pd.DataFrame:
    """Pre-registered H1-H4 on a pooled loss frame with columns named by method."""
    rows, pvals = {}, {}
    for name, lower, higher in HYPOTHESES:
        stat, p_two = dm_test(losses[lower], losses[higher])
        pvals[name] = one_sided_p(stat, p_two)
        rows[name] = {"lower": lower, "higher": higher, "dm_stat": stat,
                      "mean_lower": losses[lower].mean(), "mean_higher": losses[higher].mean()}
    table = pd.DataFrame.from_dict(rows, orient="index")
    return table.join(holm(pvals, level))
