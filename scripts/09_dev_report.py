import numpy as np
import pandas as pd

from volgate.combine.data import load_asset
from volgate.config import REPO_ROOT, load_config
from volgate.evaluate.losses import pooled_average
from volgate.evaluate.stats import dm_test
from volgate.risk.dist import alpha_tag
from volgate.risk.fz import fz0_loss

START, END = "2023-01-01", "2026-09-30"
cfg = load_config()
P = {k: REPO_ROOT / v for k, v in cfg["paths"].items()}
rows = []
for a in cfg["alphas"]:
    t = alpha_tag(a)
    per = {}
    for asset in cfg["assets"]:
        df = load_asset(P["processed"], asset).loc[START:END]
        r1 = pd.read_csv(P["processed"] / "r1" / f"{asset}.csv", index_col="date",
                         parse_dates=True).loc[START:END]
        d = df.index.intersection(r1.dropna(subset=[f"var_{t}"]).index)
        y = df.loc[d, "log_return"].to_numpy()
        v0, e0 = df.loc[d, f"har__var_{t}"].to_numpy(), df.loc[d, f"har__es_{t}"].to_numpy()
        v1, e1 = r1.loc[d, f"var_{t}"].to_numpy(), r1.loc[d, f"es_{t}"].to_numpy()
        L = pd.DataFrame({"r0": fz0_loss(y, v0, e0, a), "r1": fz0_loss(y, v1, e1, a)}, index=d)
        per[asset] = L
        s, p = dm_test(L["r1"], L["r0"])
        rows.append({"alpha": a, "asset": asset, "n": len(d), "fz0_r0": L.r0.mean(),
                     "fz0_r1": L.r1.mean(), "dm_stat": s, "dm_p": p,
                     "hit_r0": np.mean(y <= v0), "hit_r1": np.mean(y <= v1),
                     "fallback_blocks_r1": int(r1.groupby("refit_date")[f"fallback_{t}"].first().sum())})
    L = pooled_average(per)
    s, p = dm_test(L["r1"], L["r0"])
    rows.append({"alpha": a, "asset": "ALL", "n": len(L), "fz0_r0": L.r0.mean(), "fz0_r1": L.r1.mean(),
                 "dm_stat": s, "dm_p": p, "hit_r0": np.nan, "hit_r1": np.nan, "fallback_blocks_r1": np.nan})
table = pd.DataFrame(rows)
table.to_csv(P["tables"] / "r0_vs_r1.csv", index=False)
print(table[table.asset == "ALL"].round(4).to_string(index=False))
