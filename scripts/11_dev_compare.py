import numpy as np
import pandas as pd

from hart.compare import loss_frames
from volgate.config import REPO_ROOT, load_config
from volgate.evaluate.stats import dm_test

REFS = ["r0", "r1", "min_score", "mean"]
cfg = load_config()
P = {k: REPO_ROOT / v for k, v in cfg["paths"].items()}
rows = []
for a in cfg["alphas"]:
    per = loss_frames(P["processed"], cfg["assets"], a)
    for asset, L in per.items():
        for m in L.columns:
            row = {"alpha": a, "asset": asset, "method": m, "n": len(L), "mean_fz0": L[m].mean()}
            for ref in REFS:
                s, p = dm_test(L[m], L[ref]) if m != ref else (np.nan, np.nan)
                row[f"dm_vs_{ref}"], row[f"p_vs_{ref}"] = s, p
            rows.append(row)
table = pd.DataFrame(rows)
table.to_csv(P["tables"] / "compare.csv", index=False)
show = table[(table.asset == "ALL") & (table.alpha == 0.025)].sort_values("mean_fz0")
print(show.drop(columns=["alpha", "asset"]).round(4).to_string(index=False))
